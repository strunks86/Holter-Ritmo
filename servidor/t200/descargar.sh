#!/bin/bash
# Baja las bases publicas de PhysioNet (bucket abierto de AWS) a BASES. Icentia11k va al final.
B=/root/holter/data/HolterIA/bases
A=/root/venv/bin/aws
mkdir -p $B
$A configure set default.s3.max_concurrent_requests 64
d() { echo "$(date +%T) inicio $2"; $A s3 sync --no-sign-request --only-show-errors s3://physionet-open/$1 $B/$2 ${3:-}; echo "$(date +%T) fin $2 $(du -sh $B/$2 | cut -f1)"; }
d mitdb/1.0.0 mitdb &
d svdb/1.0.0 svdb &
d incartdb/1.0.0 incartdb &
d afdb/1.0.0 afdb &
d ltafdb/1.0.0 ltafdb &
d ltdb/1.0.0 ltdb &
d edb/1.0.0 edb &
d sddb/1.0.0 sddb &
d shdb-af/1.0.1 shdb-af &
d cpsc2021/1.0.0 cpsc2021 "--exclude *matlab_entry* --exclude *python_entry*" &
d challenge-2020/1.0.2/training/cpsc_2018 cpsc2018 &
d ptb-xl/1.0.3 ptb-xl "--exclude records100/*" &
for i in $(seq -w 0 10); do d icentia11k-continuous-ecg/1.0/p$i icentia11k/p$i & done
wait
echo "$(date +%T) TODO_OK"
