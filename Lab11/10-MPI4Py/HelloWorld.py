#*******************************************************************************
#
#  HelloWorld.py -
#
#   Notes:          To execute it:
#                     mpiexec -np <numprocs> python HelloWorld.py
#
#*******************************************************************************

from mpi4py import MPI
import sys

p = MPI.COMM_WORLD.Get_size ()
my_rank = MPI.COMM_WORLD.Get_rank ()
name = MPI.Get_processor_name ()

sys.stdout.write ( "Hello, World! I am process %d of %d running on %s.\n"
                    % ( my_rank, p, name) )
