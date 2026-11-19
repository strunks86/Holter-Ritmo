#!/bin/bash
B=/root/holter/data/HolterIA/bases2; cd $B
A=/root/venvdl/bin/aws
for d in chfdb stdb excluded; do
  v=$($A s3 ls --no-sign-request s3://physionet-open/$d/ | awk '/PRE/{print $2}' | tail -1)
  nice $A s3 sync --no-sign-request --only-show-errors --exclude "*.zip" s3://physionet-open/$d/$v $B/$d/
  echo "$(date +%T) $d rc=$? $(du -sh $B/$d | cut -f1)"
done
echo "$(date +%T) FIN"
