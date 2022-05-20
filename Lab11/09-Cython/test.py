#*******************************************************************************
#
#  test.py - A Python script that uses the sqax extension module
#
#   Notes:
#
#*******************************************************************************

import sys
# Import the extension module xsqax.
import xsqax

if len ( sys.argv ) != 2:
  print ( '\nUsage: python {} x\n'.format ( sys.argv[0] ) )
  exit ()
else:
  # Call the print_result method in the xsqax extension
  x = float ( sys.argv[1] )
  xsqax.print_result ( x )
