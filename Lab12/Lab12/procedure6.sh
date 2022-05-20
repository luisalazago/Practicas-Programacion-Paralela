#!/bin/bash
#
# procedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 matSide\n\t(e.g., $0 16384)\n"
  exit 1
else
  echo -e "Will run MatMatAdd5.py for profiling purposes using the memory_profiler module\n"
  python -m memory_profiler MatMatAdd5.py $1 > MatMatAdd5_mprof.out
  echo -e "\nDone.\n"
fi
