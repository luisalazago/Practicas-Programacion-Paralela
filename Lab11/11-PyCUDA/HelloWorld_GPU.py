#*******************************************************************************
#
#  HelloWorld_GPU.py -  The classical Hello World using PyCUDA
#
#  Notes:               Assumes Python 3.8
#
#*******************************************************************************

from os import system
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

system ( 'clear' )

kernel = SourceModule \
('''
#include <stdio.h>

__global__ void hello_world_krnl ()
{
  printf ( "Hello world from thread %d, in block %d!\\n", threadIdx.x, blockIdx.x );

  __syncthreads ();

  if ( threadIdx.x == 0 && blockIdx.x == 0 )
  {
      printf ( "-------------------------------------\\n");
      printf ( "This kernel was launched over a grid consisting of %d blocks,\\n", gridDim.x );
      printf ( "where each block has %d threads.\\n\\n", blockDim.x );
  }
}
''')

hello_krnl = kernel.get_function ( "hello_world_krnl" )
hello_krnl ( block = ( 5, 1, 1 ), grid = ( 2, 1, 1 ) )
