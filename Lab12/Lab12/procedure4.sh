#!/bin/bash
#
# procedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 matSide\n\t(e.g., $0 16384)\n"
  exit 1
else
  echo -e "Will run MatMatAdd3.py for profiling purposes using the memory_profiler module. It takes a while, so be patient please...\n"
  python -m memory_profiler MatMatAdd3.py $1 > MatMatAdd3_mprof.out
  echo -e "\nDone.\n"
fi
