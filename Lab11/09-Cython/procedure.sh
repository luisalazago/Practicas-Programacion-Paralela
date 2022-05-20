#!/bin/bash
#
# CythonProcedure.sh
#

clear

if [ "$#" -ne 1 ]; then
  echo -e "Usage: $0 x\n"
  exit 1
else
  # Build 'xsqax' extension:
  #   xsqax.c
  #   build directory
  #   xsqax.cpython-38-x86_64-linux-gnu.so
  #
  echo -e "Building sxqax extension...\n"
  python3 setup.py build_ext --inplace

  # Delete unnecessary files
  #
#  rm -fR build
 # rm -f xsqax.c

  # Test the extension
  #
  echo -e "\nTesting the sxqax extension with x = $1...\n"
  python3 test.py $1
  echo -e "\nDone.\n"
fi
