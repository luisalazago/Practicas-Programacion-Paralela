#*******************************************************************************
#
#  multiprocessing_test.py -  A Python script that invokes the do_something
#                             function to "process" size elements using numProcs
#                             processes
#
#   Notes:                  The number of processes, numProcs, is equal to the
#                           number of iterations, numExecutions, in the serial
#                           (sequential) version
#
#*******************************************************************************

import multiprocessing
from do_something import *
import time

if __name__ == "__main__":

  tstart = time.time ()

  size = 10000000
  numProcs = 10
  jobs = []

  # create the processes
  for p in range ( 0, numProcs ):
    out_list = list ()
    process = multiprocessing.Process\
              ( target = do_something, args = ( size, out_list ) )
    jobs.append ( process )

  # start the processes
  for j in jobs:
    j.start ()

  # wait for the processes to finish
  for j in jobs:
    j.join ()

  print ("List processing complete.")

  tend = time.time ()
  print ( "multiprocesses time=", tend - tstart )
