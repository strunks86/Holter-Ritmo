#!/bin/bash
# T203 no supero a T202 (mas Icentia no ayuda). T204: robustez a orden/polaridad de canales, desde T202.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
timeout 900 $P t202_train.py --name smoke204 --init runs/t202/best.pt --rob 0.5 --minutes 1 --every 0.5 --val_cap 1500 --workers 2 > $L/smoke204.log 2>&1
grep -q "FIN pasos" $L/smoke204.log || { log "T204: prueba corta fallo; sigue T203"; exit 1; }
pkill -f "t200/noche3.sh"; sleep 1; pkill -f "name t203"; sleep 5
log "T203 detenido (no supero a T202). inicio T204 (robustez de canales) desde runs/t202/best.pt"
$P t202_train.py --name t204 --init /root/holter/t200/runs/t202/best.pt --tags holter,icentia,taqui,icentia2 --rob 0.5 --lr_bb 1e-5 --lr_head 5e-5 --minutes 600 --every 30 --paciencia 6 --workers 6 > $L/train_t204.log 2>&1
log "T204 entrenamiento rc=$?"
