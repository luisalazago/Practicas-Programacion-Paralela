#!/bin/bash

#
# lgprof_proc.sh
#

clear

echo -e "Will run sort for line profiling purposes using gprof\n"
gcc -Wall -pg -g sort.c -o sort_gl
./sort_gl
gprof -l -b sort_gl | less
gprof -l sort_gl | gprof2dot | dot -Tpng -o sort_lgprof.png; cat sort_lgprof.png | display
echo -e "\nDone.\n"
