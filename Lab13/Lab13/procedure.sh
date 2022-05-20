#!/usr/bin/env bash
#
# procedure.sh
#

clear

if [ "$#" -ne 2 ]; then
  echo -e "Usage: $0 matSide blkSide\n\t(e.g., $0 2048 16)\n"
  exit 1
else
  echo -e "Will build MatMatMult9 for profiling purposes using gprof\n"
  make
  echo -e "\nDone.\n"
  echo -e "Will run MatMatMult9 for profiling purposes using gprof\n"
  ./MatMatMult9 $1 $2
  echo -e "\nDone.\n"
  echo -e "Will visualise results using Image Magick's display\n"
  gprof MatMatMult9 | gprof2dot | dot -Tpng -o MMM9_gprof.png ; cat MMM9_gprof.png | display
  echo -e "\nDone.\n"
fi
