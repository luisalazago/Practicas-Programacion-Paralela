#*******************************************************************************
#
#  MatMatAdd4.py - A program to add two matrices using NumPy
#
#   Notes:          Matrices are wired and have integer elements
#
#*******************************************************************************

import sys
import numpy as np

if len(sys.argv) != 2:
  print ( '\nUsage: python {} N\nwhere N is the number of matrix rows and cols\n'.format(sys.argv[0]) )
  exit ()
else:
  S = int (sys.argv[1])
  Rows = S
  Cols = S
  print ( f'\nAdding {Rows}x{Cols} matrices' )

# Matrix shape is NxN ( N, N )
MA = np.full ( ( Rows, Cols ), 1.0, dtype = float )
MB = np.full ( ( Rows, Cols ), 2.0, dtype = float )

MY = MA + MB

Rows, Cols = MY.shape
if Rows <= 32:
  print ( MY )
