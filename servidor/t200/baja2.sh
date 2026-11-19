#!/bin/bash
# Bases nuevas (19-nov, busqueda pedida por Jorge)
B=/root/holter/data/HolterIA/bases2; mkdir -p $B; cd $B
A=/root/venvdl/bin/aws
for d in vfdb cudb leipzig-heart-center-ecg butqdb ecg-arrhythmia afpdb aftdb pwave macecgdb nsrdb but-pdb; do
  v=$($A s3 ls --no-sign-request s3://physionet-open/$d/ | awk '/PRE/{print $2}' | tail -1)
  echo "$(date +%T) $d $v"
  nice $A s3 sync --no-sign-request --only-show-errors --exclude "*.zip" s3://physionet-open/$d/$v $B/$d/ 
  echo "$(date +%T) $d rc=$? $(du -sh $B/$d | cut -f1)"
done
mkdir -p noise-holter-database && cd noise-holter-database
wget -q -c https://physionet.org/content/noise-holter-database/get-zip/1.0.0/ -O nhdb.zip; echo "$(date +%T) nhdb rc=$? $(ls -la nhdb.zip)"
unzip -q -o nhdb.zip; cd ..
mkdir -p cpsc2020 && cd cpsc2020
wget -q -c --tries=3 --timeout=60 https://opensz.oss-cn-beijing.aliyuncs.com/ICBEB2020/file/TrainingSet.zip; echo "$(date +%T) cpsc2020 rc=$? $(ls -la)"
unzip -q -o TrainingSet.zip; cd ..
mkdir -p iridia-af && cd iridia-af
wget -q -O rec.json "https://zenodo.org/api/records/8405941"; python3 -c "
import json;d=json.load(open('rec.json'))
for f in d.get('files',[]): print(f['links']['self'], f['key'], f['size'])" > files.txt; cat files.txt
while read u k s; do wget -q -c -O "$k" "$u"; echo "$(date +%T) $k rc=$?"; done < files.txt
echo "$(date +%T) FIN"; du -sh $B/*
