#!/bin/bash

#
# valgrind_proc.sh
#

clear

echo -e "Will run sort for line profiling purposes using valgrind. It takes a while, so be patient please\n"
gcc -Wall -pg sort.c -o sort_v
valgrind --tool=callgrind ./sort_v
ogg123 -y 2 -q complete.oga
kcachegrind &
echo -e "\nDone.\n"
