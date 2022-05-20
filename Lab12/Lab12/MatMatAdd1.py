#*******************************************************************************
#
#  MatMatAdd1.py - A program to add two matrices using nested loops
#
#   Notes:          Matrices are wired and have integer elements
#                   Uses a straightforward approach
#
#*******************************************************************************

import sys
from time import time

#===( Setup )==============================================================

if len(sys.argv) != 2:
  print ( '\nUsage: python {} N\nwhere N is the number of matrix rows and cols\n'.format(sys.argv[0]) )
  exit ()
else:
  S = int (sys.argv[1])
  Rows = S
  Cols = S
  print ( f'\nAdding {Rows}x{Cols} matrices' )

#===( Initialisation )=====================================================

# Initialize matrix A
matA = []

for r in range ( Rows ):      # A for loop for row entries
  v = []
  for c in range ( Cols ):  # A for loop for column entries
    v.append ( 1.0 )
  matA.append ( v )

# Initialize matrix B
matB = []

for r in range ( Rows ):      # A for loop for row entries
  v = []
  for c in range ( Cols ):  # A for loop for column entries
    v.append ( 2.0 )
  matB.append ( v )

# Initialize matrix Y
matY = []

for r in range ( Rows ):      # A for loop for row entries
  v = []
  for c in range ( Cols ):  # A for loop for column entries
    v.append ( 0.0 )
  matY.append ( v )


#===( Processing )=========================================================

# iterating by rows
for row in range ( Rows ):

  # iterating by columns
  for col in range ( Cols ):

      matY[row][col] = matA[row][col] + matB[row][col]


#===( Visualisation )======================================================

if S <= 32:
  # Print resulting matrix
    print ( '\nY =' )
    for r in range ( Rows ):
      for c in range ( Cols ):
        print ( matY[r][c], end = ' ' )
      print ()

