#!/bin/bash
#
# procedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 matSide\n\t(e.g., $0 16384)\n"
  exit 1
else
  echo -e "Will run MatMatAdd1.py for profiling purposes using the cProfile module\n"
  python -m cProfile -s cumulative MatMatAdd1.py $1 > MatMatAdd1_cprof.out
  echo -e "\nDone.\n"
  echo -e "Will run MatMatAdd1.py for profiling purposes using the cProfile module. The result will be visualised via SnakeViz\n"
  python -m cProfile -o MatMatAdd1.cprof MatMatAdd1.py $1
  echo -e "\nDone.\n"
  echo -e "Will run SnakeViz to visualise results using a web browser\n"
  snakeviz MatMatAdd1.cprof &
  echo -e "\nDone.\n"
fi
