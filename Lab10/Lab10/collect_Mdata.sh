#!/bin/bash

#
# collect_data.sh
#

# get host name
hname=$(hostname)
# get processor name
proc=$(lscpu | grep "Model name" | cut -d: -f2 | sed -e 's/[ "()RTM]//g')
dim0=4
dimN=512

# clear out any previous versions of files we will build
rm -f MMM-${hname}-${proc}.out

# sweep through a range of matrix dimensions
for (( s = dim0; s <= dimN; s *= 2 ))
  do
    python MatMatMult1_t.py $s | tee tmp.$$
    echo -e "$s x $s matrices:\n`cat tmp.$$ | grep "It" | awk '{print $3}'`" >> MMM-${hname}-${proc}.out
  done

# clean up
rm -f tmp.$$
