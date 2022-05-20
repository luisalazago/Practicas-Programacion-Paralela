#!/bin/bash
#
# procedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 matSide\n\t(e.g., $0 16384)\n"
  exit 1
else
  echo -e "Will run the kernprof script to instrument the code in MatMatAdd5.py for profiling purposes using the line_profiler module\n"
  kernprof -l -v MatMatAdd5.py $1
  echo -e "\nDone.\n"
  echo -e "Will run MatMatAdd5.py.lprof to visualise the profiling results\n"
  python -m line_profiler MatMatAdd5.py.lprof
  echo -e "\nDone.\n"
fi
