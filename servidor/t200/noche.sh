#!/bin/bash
# Fase noche: T201 = mejor T200 + salida RUIDO, todas las bases, hasta ~07:40 Chile (10:40 UTC); luego pruebas.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
until grep -q "evaluacion rc" $L/orquesta.log; do sleep 20; done
until [ -f data/icentia_meta.npz ] || grep -q "build icentia rc" $L/orquesta.log; do sleep 20; done
log "ventanas de icentia listas"
[ -f runs/t200/best.pt ] || cp runs/t200/last.pt runs/t200/best.pt
until [ "$(ls /root/holter/data/HolterIA/bases/nstdb/{em,ma,bw}.dat 2>/dev/null | wc -l)" = 3 ]; do sleep 10; done
MIN=$(( ( $(date -d "10:40" +%s) - $(date +%s) ) / 60 ))
log "inicio noche T201 (con ruido) $MIN min desde runs/t200/best.pt"
$P t201_train.py --name noche --init /root/holter/t200/runs/t200/best.pt --tags holter,icentia --minutes $MIN --every 30 --bs 64 --workers 8 > $L/train_noche.log 2>&1
log "noche entrenamiento rc=$?"
$P t201_eval.py --t201 /root/holter/t200/runs/noche/best.pt --name ruido > $L/eval_ruido.log 2>&1
log "noche evaluacion ruido rc=$?"
