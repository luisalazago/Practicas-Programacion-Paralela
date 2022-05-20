#!/bin/bash

#
# gprof_proc.sh
#

clear

echo -e "Will run sort for profiling purposes using gprof\n"
gcc -Wall -pg sort.c -o sort_g
./sort_g
gprof -q -b sort_g
gprof -p -b sort_g
gprof sort_g | gprof2dot | dot -Tpng -o sort_gprof.png; cat sort_gprof.png | display
echo -e "\nDone.\n"
