#*******************************************************************************
#
#  do_something.py -  A Python script that builds a list of pseudo-random
#                     integers (the "workload")
#
#   Notes:
#
#*******************************************************************************

import random

def do_something ( count, out_list ):
  for i in range (count):
    out_list.append ( random.random() )
