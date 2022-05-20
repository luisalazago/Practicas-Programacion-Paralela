#*******************************************************************************
#
#  MatMatAdd3.py -    A Python script to add two matrices using PyCUDA
#
#   Notes:
#
#*******************************************************************************

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import numpy

size = 4

# Initialise A, B matrices
MA_gpu = gpuarray.to_gpu ( numpy.random.randn ( size, size ).astype ( numpy.float32 ) )
MB_gpu = gpuarray.to_gpu ( numpy.random.randn ( size, size ).astype ( numpy.float32 ) )

MY = ( MA_gpu + MB_gpu ).get ()

# Print matrices
print ( "MA =" )
print ( MA_gpu )
print ( "MB =" )
print ( MB_gpu )
print ( "MY =" )
print ( MY )
