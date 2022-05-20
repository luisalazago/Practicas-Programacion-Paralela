#*******************************************************************************
#
#  checkDeviceInfo.py - Display a variety of information on the first CUDA
#                       device in this system, including driver version,
#                       runtime version, compute capability, bytes of global
#                       memory, etc.
#
#  Notes:               Assumes Python 3.8
#
#*******************************************************************************

from os import system
import pycuda
import pycuda.driver as CUDA

system ( 'clear' )

CUDA.init ()

print ( 'checkDeviceInfo Starting... (PyCUDA version)\n' )

print ( 'Detected {} CUDA Capable device(s) \n'.format ( CUDA.Device.count() ) )

for deviceNum in range ( CUDA.Device.count() ):

  GPU_device = CUDA.Device ( deviceNum )
  print ( 'Device {}: "{}"'.format( deviceNum, GPU_device.name () ) )
  compute_capability = float ( '%d.%d' % GPU_device.compute_capability () )
  print ( '  Compute Capability: {}'.format ( compute_capability) )
  print ( '  Total Memory: {} mebibytes'.format ( GPU_device.total_memory() // (1024 ** 2) ) )

  # The following will give us all remaining device attributes as seen
  # in the original deviceQuery.

  device_attributes_tuples = GPU_device.get_attributes().items()
  device_attributes = {}

  for k, v in device_attributes_tuples:
      device_attributes[str(k)] = v

  num_mp = device_attributes['MULTIPROCESSOR_COUNT']

  # Cores per multiprocessor is not reported by the GPU!
  # We must use a lookup table based on compute capability.
  # See the following:
  # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

  cuda_cores_per_mp = { 3.5 : 192, 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128, 7.5 : 64 }[compute_capability]

  print ( '  {} Multiprocessors, {} CUDA Cores / Multiprocessor: {} CUDA Cores\n'.format ( num_mp, cuda_cores_per_mp, num_mp * cuda_cores_per_mp ) )

  device_attributes.pop('MULTIPROCESSOR_COUNT')

  system ( 'read -p "Press <Enter> to continue..."' )

  # Print all remaining device attributes as seen in the original deviceQuery.

  print ( '\nFull list of device attributes:\n' )
  for k in device_attributes.keys():
    print ( '\t{}: {}'.format(k, device_attributes[k]) )
