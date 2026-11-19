#!/bin/bash
# Cambio a T202 (TV y TSV) cuando esten listas las etiquetas; sigue hasta ~07:40 Chile y prueba todo.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
until grep -qE "ETIQUETAS_OK|Traceback" $L/taqui.log; do sleep 20; done
grep -q ETIQUETAS_OK $L/taqui.log || { log "T202: fallo armando etiquetas; sigue T201"; exit 1; }
log "etiquetas TV/TSV listas; prueba corta de T202"
timeout 900 $P t202_train.py --name smoke7 --minutes 1 --every 0.5 --val_cap 2000 --workers 2 > $L/smoke7.log 2>&1
grep -q "^.*FIN pasos" $L/smoke7.log || { log "T202: la prueba corta fallo; sigue T201"; exit 1; }
pkill -f "t200/noche.sh"; sleep 1; pkill -f t201_train.py; sleep 5
cp runs/noche/best.pt runs/noche/best_t201.pt
MIN=$(( ( $(date -d "10:40" +%s) - $(date +%s) ) / 60 ))
[ $MIN -lt 60 ] && MIN=300
log "inicio T202 (TV y TSV) $MIN min desde runs/noche/best_t201.pt"
$P t202_train.py --name t202 --init /root/holter/t200/runs/noche/best_t201.pt --minutes $MIN --every 30 --workers 8 > $L/train_t202.log 2>&1
log "T202 entrenamiento rc=$?"
$P t202_eval.py --modelos "T158=/root/holter/t158/runs/full_s0/best.pt,T200=/root/holter/t200/runs/t200/best.pt,T201=/root/holter/t200/runs/noche/best_t201.pt,T202=/root/holter/t200/runs/t202/best.pt" --name final > $L/eval_final.log 2>&1
log "T202 evaluacion final rc=$?"
