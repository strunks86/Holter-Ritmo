#!/bin/bash
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
until [ -f /root/holter/t158/data/windows_meta.npz ]; do sleep 10; done
log "inicio entrenamiento 60 min"
$P t200_train.py --name t200 --tags holter,icentia --minutes 60 --every 10 --bs 64 --workers 10 > $L/train.log 2>&1
log "entrenamiento rc=$?"
$P t200_eval.py --cand /root/holter/t200/runs/t200/best.pt --name t200 --tags holter,icentia > $L/eval.log 2>&1
log "evaluacion rc=$?"
