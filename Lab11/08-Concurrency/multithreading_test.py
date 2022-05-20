#*******************************************************************************
#
#  multithreading_test.py - A Python script that invokes the do_something
#                           function to "process" size elements using numThreads
#                           threads
#
#   Notes:                  The number of threads, numThreads, is equal to the
#                           number of iterations, numExecutions, in the serial
#                           (sequential) version
#
#*******************************************************************************

import threading
from do_something import *
import time

if __name__ == "__main__":

  tstart = time.time ()

  size = 10000000
  numThreads = 10
  jobs = []

  # create the threads
  for t in range ( 0, numThreads ):
    out_list = list ()
    thread = threading.Thread ( target = do_something ( size, out_list ) )
    jobs.append ( thread )

  # start the threads
  for j in jobs:
    j.start ()

  # wait for the threads to finish
  for j in jobs:
    j.join ()

  print ( "List processing complete." )

  tend = time.time ()
  print ( "multithreading time=", tend - tstart )
