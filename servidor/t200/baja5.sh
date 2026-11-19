#!/bin/bash
# Long-Term ST por archivos sueltos, 8 en paralelo (el zip quedo cortado)
cd /root/holter/data/HolterIA/bases2/ltstdb || exit 1
B=https://physionet.org/files/ltstdb/1.0.0
wget -q -c $B/RECORDS -O RECORDS
for r in $(cat RECORDS); do for e in hea dat atr; do echo $B/$r.$e; done; done > urls.txt
xargs -P 8 -n 1 wget -q -c --tries=5 < urls.txt
echo "$(date +%T) FIN $(ls *.dat | wc -l) dat $(du -sh . | cut -f1)"
