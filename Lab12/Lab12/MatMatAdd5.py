#*******************************************************************************
#
#  MatMatAdd5.py - A program to add two matrices using NumPy
#
#   Notes:          Matrices are wired and have integer elements
#
#*******************************************************************************

import sys
import numpy as np

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

  # Matrix shape is NxN ( N, N )
  matA = np.full ( ( Rows, Cols ), 1.0, dtype = float )
  matB = np.full ( ( Rows, Cols ), 2.0, dtype = float )

  return ( matA, matB, matY )

#===( Processing )=========================================================

@profile
def process ( matA, matB, matY ):

  matY = matA + matB
  return ( matY )

#===( Visualisation )======================================================

def visualise ( matY ):

  Rows, Cols = MY.shape
  if Rows <= 32:
    print ( MY )

#===( Main function )======================================================

def main ():

  S, Rows, Cols = setup ()

  matA = []
  matB = []
  matY = []
  matA, matB, matY = initialise ( matA, matB, matY, Rows, Cols )

  matY = process ( matA, matB, matY )
  if S <= 32:
    visualise ( matY )

if __name__ == "__main__":
  main ()
