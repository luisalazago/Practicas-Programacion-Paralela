#*******************************************************************************
#
#  MatMatAdd1.py -    A Python script to add two matrices using PyCUDA
#
#   Notes:
#
#*******************************************************************************

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

kernel = SourceModule \
("""
  __global__ void MatAdd ( float *MA, float *MB, float *MY )
  {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    MY[row * 4 + col] = MA[row * 4 + col] + MB[row * 4 + col];
  }
""")

# matrix side
size = 4

# Initialise A, B matrices
MA = numpy.random.randn ( size, size )
MA = MA.astype ( numpy.float32 )

MB = numpy.random.randn ( size, size )
MB = MB.astype ( numpy.float32 )

# Allocate device memory
d_MA = cuda.mem_alloc ( MA.size * MA.dtype.itemsize )
d_MB = cuda.mem_alloc ( MB.size * MB.dtype.itemsize )

MY = numpy.empty_like ( MA )
d_MY = cuda.mem_alloc ( MY.size * MY.dtype.itemsize )

# Copy matrices to device
cuda.memcpy_htod ( d_MA, MA )
cuda.memcpy_htod ( d_MB, MB )

# Launch kernel
MatAdd_GPU = kernel.get_function ( "MatAdd" )
MatAdd_GPU ( d_MA, d_MB, d_MY, block = ( size, size, 1 ) )

# Copy result matrix from device
cuda.memcpy_dtoh ( MY, d_MY )

# Print matrices
print ( "MA =" )
print ( MA )
print ( "MB =" )
print ( MB )
print ( "MY =" )
print ( MY )
