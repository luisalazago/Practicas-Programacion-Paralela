#*******************************************************************************
#
#  MatMatAdd3.py - A program to add two matrices using nested loops
#
#   Notes:          Matrices are wired and have integer elements
#                   Uses UDFs (User-Defined Functions) and a straightforward
#                   approach
#
#                   It is identical to MatMatAdd2.py; adds @profile decorator
#
#*******************************************************************************

import sys

#===( Setup )==============================================================

def setup ():

  if len(sys.argv) != 2:
    print ( '\nUsage: python {} N\nwhere N is the number of matrix rows and cols\n'.format(sys.argv[0]) )
    exit ()
  else:
    S = int (sys.argv[1])
    Rows = S
    Cols = S
    print ( f'\nAdding {Rows}x{Cols} matrices' )
    return ( S, Rows, Cols )

#===( Initialisation )=====================================================

@profile
def initialise ( matA, matB, matY, Rows, Cols ):

  # Initialize matrix A

  for r in range ( Rows ):      # A for loop for row entries
    v = []
    for c in range ( Cols ):  # A for loop for column entries
      v.append ( 1.0 )
    matA.append ( v )

  # Initialize matrix B
  for r in range ( Rows ):      # A for loop for row entries
    v = []
    for c in range ( Cols ):  # A for loop for column entries
      v.append ( 2.0 )
    matB.append ( v )

  # Initialize matrix Y
  for r in range ( Rows ):      # A for loop for row entries
    v = []
    for c in range ( Cols ):  # A for loop for column entries
      v.append ( 0.0 )
    matY.append ( v )

  return ( matA, matB, matY )

#===( Processing )=========================================================

def process ( matA, matB, matY, Rows, Cols ):

  # iterating by rows
  for row in range ( Rows ):

    # iterating by columns
    for col in range ( Cols ):

        matY[row][col] = matA[row][col] + matB[row][col]

  return ( matY )


#===( Visualisation )======================================================

def visualise ( matY, Rows, Cols ):

  # Print resulting matrix
    print ( '\nY =' )
    for r in range ( Rows ):
      for c in range ( Cols ):
        print ( matY[r][c], end = ' ' )
      print ( '' )
    print ( '' )

#===( Main function )======================================================

def main ():

  S, Rows, Cols = setup ()

  matA = []
  matB = []
  matY = []
  matA, matB, matY = initialise ( matA, matB, matY, Rows, Cols )

  matY = process ( matA, matB, matY, Rows, Cols )
  if S <= 32:
    visualise ( matY, Rows, Cols )

if __name__ == "__main__":
  main ()
