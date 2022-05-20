#!/bin/bash
#
# procedure.sh
#

clear
# get host name
hname=$(hostname)
# get processor name
proc=$(lscpu | grep "Model name" | cut -d: -f2 | sed -e 's/["()RTM]//g')
# get Python version
pythonver=$(python --version)

# run tests
echo -e "Running scripts using ${pythonver} on ${hname} on a ${proc}. Please wait...\n"
python3 serial_test.py
echo ""
python3 multithreading_test.py
echo ""
python3 multiprocessing_test.py
echo -e "\nDone.\n"
