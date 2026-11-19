#!/bin/bash
# T202 se estanco (sobreajuste): prueba final de T202 y luego T203 con 238.000 ventanas mas de Icentia.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
pkill -f "t200/noche2.sh"; sleep 1; pkill -f t202_train.py; sleep 5
log "T202 detenido por estancamiento (mejor = min 120); prueba final"
( $P t202_eval.py --modelos "T158=/root/holter/t158/runs/full_s0/best.pt,T200=/root/holter/t200/runs/t200/best.pt,T201=/root/holter/t200/runs/noche/best_t201.pt,T202=/root/holter/t200/runs/t202/best.pt" --name final > $L/eval_final.log 2>&1; log "evaluacion final T202 rc=$?" ) &
log "inicio T203 (mas Icentia, lr mas baja) desde runs/t202/best.pt"
$P t202_train.py --name t203 --init /root/holter/t200/runs/t202/best.pt --tags holter,icentia,taqui,icentia2 --lr_bb 1e-5 --lr_head 5e-5 --minutes 600 --every 30 --paciencia 6 --workers 6 > $L/train_t203.log 2>&1
log "T203 entrenamiento rc=$?"
