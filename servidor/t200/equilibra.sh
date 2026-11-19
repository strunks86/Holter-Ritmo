#!/bin/bash
# Prioridad: 1) terminar ventanas T158  2) entrenar en GPU  3) ventanas nuevas y descarga en segundo plano (nice 19)
L=/root/holter/t200/logs/orquesta.log
pkill -STOP -f "t200_build.py"
echo "$(date +%T) equilibra: pauso build holter y descarga hasta tener ventanas T158" >> $L
until [ -f /root/holter/t158/data/windows_meta.npz ]; do sleep 5; done
until pgrep -f t200_train.py >/dev/null; do sleep 5; done
sleep 60
for p in $(pgrep -f "t200_build.py|aws s3 sync"); do renice -n 19 -p $p >/dev/null; done
pkill -CONT -f "t200_build.py"; pkill -CONT -f "aws s3 sync"
echo "$(date +%T) equilibra: entrenamiento en marcha; reanudo build holter y descarga con nice 19" >> $L
