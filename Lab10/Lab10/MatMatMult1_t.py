#*******************************************************************************
#
#  MatMatMult1.py - A program to multiply two matrices using nested loops
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

#===( Initialisation )=====================================================

tstart = time ()

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

tstop = time ()

init_time = tstop - tstart

#===( Processing )=========================================================

tstart = time ()

# iterating by rows of matA
for row in range ( Rows ):

  # iterating by columns of matB
  for col in range ( Cols ):

    # iterating by rows of matB
    for i in range ( S ):
      matY[row][col] += matA[row][i] * matB[i][col]

tstop = time ()

proc_time = tstop - tstart

#===( Visualisation )======================================================

if S <= 32:
  tstart = time ()

# Print resulting matrix
  print ( '\nY =' )
  for r in range ( Rows ):
    for c in range ( Cols ):
      print ( matY[r][c], end = ' ' )
    print ()

  tstop = time ()

  dump_time = tstop - tstart

#===( Execution Report )===================================================

print ( f'\nMultiplying {Rows}x{Cols} matrices' )
print ( 'It took {:16.12f} seconds to initialise matrices'.format(init_time) )
print ( 'It took {:16.12f} seconds to process matrices'.format(proc_time) )
if S <= 16:
  print ( 'It took {:16.12f} seconds to print result matrix'.format(dump_time) )
print ()
