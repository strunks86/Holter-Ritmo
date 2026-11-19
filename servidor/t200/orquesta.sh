#!/bin/bash
# Arma ventanas y entrena 60 min. Icentia entra al entrenamiento en caliente cuando este lista.
set -u
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs; mkdir -p $L
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
B=/root/holter/data/HolterIA/bases
until $P -c "import wfdb,onnx,onnxruntime" 2>/dev/null; do sleep 10; done
until grep -q "fin ptb-xl" descargar.log; do sleep 10; done
log "ptb-xl listo (cpsc2018 fuera: physionet.org va muy lento)"
( $P $HOLTERAI/scripts/t158_build.py --workers 14 > $L/build_t158.log 2>&1; log "t158_build rc=$?" ) &
( $P t200_build.py --workers 14 --tag holter --dbs mitdb,svdb,incartdb,ltdb,edb,sddb,shdb-af,cpsc2021 > $L/build_holter.log 2>&1; log "build holter rc=$?" ) &
( until grep -q TODO_OK descargar.log; do sleep 20; done; log "icentia descargada";
  $P t200_build.py --workers 20 --tag icentia --dbs icentia11k > $L/build_icentia.log 2>&1; log "build icentia rc=$?" ) &
until [ -f /root/holter/t158/data/windows_meta.npz ] && [ -f data/holter_meta.npz ]; do sleep 10; done
log "inicio entrenamiento 60 min"
$P t200_train.py --name t200 --tags holter,icentia --minutes 60 --every 10 --bs 64 --workers 10 > $L/train.log 2>&1
log "entrenamiento rc=$?"
$P t200_eval.py --cand /root/holter/t200/runs/t200/best.pt --name t200 --tags holter,icentia > $L/eval.log 2>&1
log "evaluacion rc=$?"
