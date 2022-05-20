#*******************************************************************************
#
#  snd_rcv.py - A program that illustrates sending/receiving messages in MPI
#               using:
#                 MPI_Send
#                 MPI_Recv
#
#   Notes:          To execute it:
#                     mpiexec -np <numprocs> python HelloWorld.py
#
#*******************************************************************************

from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank ()

# a float64 array of a single element
randNum = numpy.zeros ( 1 )

if my_rank == 0:
  randNum = numpy.random.random_sample ( 1 )
  print ( "proc", my_rank, ": ", randNum[0] )
  comm.Send ( randNum, dest = 1 )
  print ( "proc", my_rank, "sent", randNum[0] )

if my_rank == 1:
  print ( "proc", my_rank, ": ", randNum[0] )
  comm.Recv ( randNum, source = 0 )
  print ( "proc", my_rank, "received", randNum[0] )
