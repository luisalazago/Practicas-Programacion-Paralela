#!/bin/bash

#
# gperf_proc.sh
#

clear

echo -e "Will run sort for profiling purposes using GPerfTools. It takes a while, so be patient please\n"
gcc -Wall -g sort.c -o sort_gperf
distrib=$(awk '/^NAME=/' /etc/*-release | cut -d= -f2 | sed -e 's/"//g')
if [ "$distrib" = "Ubuntu" ]; then
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libprofiler.so CPUPROFILE=sort_gperf.prof ./sort_gperf
  google-pprof --web ./sort_gperf ./sort_gperf.prof &
else
  LD_PRELOAD=/usr/lib64/libprofiler.so CPUPROFILE=sort_gperf.prof ./sort_gperf
  pprof --web ./sort_gperf ./sort_gperf.prof &
fi
ogg123 -q -y 2 complete.oga
echo -e "\nDone.\n"
