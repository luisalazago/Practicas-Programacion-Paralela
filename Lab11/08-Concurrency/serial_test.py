#*******************************************************************************
#
#  serial_test.py - A Python script that invokes the do_something function
#                   numExecutions times to "process" size elements
#
#   Notes:
#
#*******************************************************************************

import time
from do_something import *

if __name__ == "__main__":

  tstart = time.time ()

  size = 10000000
  numExecutions = 10

  for i in range ( 0, numExecutions ):
    out_list = list ()
    do_something ( size, out_list )

  print ( "List processing complete." )

  tend = time.time ()
  print ( "serial time = ", tend - tstart )
