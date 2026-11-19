#!/bin/bash
B=/root/holter/data/HolterIA/bases2; cd $B
A=/root/venvdl/bin/aws
for d in mghdb ucddb challenge-2017 challenge-2011; do
  v=$($A s3 ls --no-sign-request s3://physionet-open/$d/ | awk '/PRE/{print $2}' | tail -1)
  nice $A s3 sync --no-sign-request --only-show-errors --exclude "*.zip" s3://physionet-open/$d/$v $B/$d/
  echo "$(date +%T) $d rc=$? $(du -sh $B/$d | cut -f1)"
done
mkdir -p ltstdb && cd ltstdb
timeout 3000 wget -q -c --tries=3 https://physionet.org/static/published-projects/ltstdb/long-term-st-database-1.0.0.zip -O ltstdb.zip; echo "$(date +%T) ltstdb rc=$? $(ls -la ltstdb.zip)"
cd ../iridia-af && ls -la && (unzip -q -o iridia-af-records-v1.0.1.zip; echo "$(date +%T) iridia unzip rc=$?"; du -sh .)
echo "$(date +%T) FIN"
