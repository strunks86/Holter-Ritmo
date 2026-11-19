#!/bin/bash
# T206: parte del promedio T204+T205b (SOPA) y elige el mejor con validacion antigua + bases nuevas.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
pkill -f "t200/noche6.sh"; sleep 1; pkill -f "name t205b"; sleep 3
log "T205b detenido (puntos min30..min120 guardados). inicio T206 desde SOPA con validacion antigua+nueva"
$P t202_train.py --name t206 --init /root/holter/t200/runs/t205b/sopa50.pt --tags $TAGS --val_tags holter,icentia,taqui,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20 --rob 0.5 --lr_bb 5e-6 --lr_head 2.5e-5 --minutes 600 --every 30 --paciencia 5 --workers 6 > $L/train_t206.log 2>&1
log "T206 entrenamiento rc=$?"
