#!/bin/bash
#
# procedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 numProcs\n"
  exit 1
else
  echo -e "Will run $1 instance processes of HelloWorld.py with the aid of mpi4py\n"
  mpiexec -np $1 python HelloWorld.py
  echo -e "\nDone.\n"
fi
