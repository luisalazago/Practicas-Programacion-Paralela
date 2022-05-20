/*
 * Display a variety of information on the first CUDA device in this system,
 * including driver version, runtime version, compute capability, bytes of
 * global memory, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cudaChk.h"

int main ( int argc, char **argv )
{
  system ( "clear" );
  printf ( "%s Starting...\n", argv[0] );

  int deviceCount = 0;

  cudaGetDeviceCount ( &deviceCount );

  if  ( deviceCount == 0 )
  {
    printf ( "There are no available device (s) that support CUDA\n" );
  }
  else
  {
      printf ( "\nDetected %d CUDA Capable device(s)\n", deviceCount );
  }

  int dev = 0, driverVersion = 0, runtimeVersion = 0;

  CHECK ( cudaSetDevice ( dev ) );

  cudaDeviceProp deviceProp;
  CHECK ( cudaGetDeviceProperties ( &deviceProp, dev ) );
  printf ( "\nDevice %d: \"%s\" (%s)\n\n", dev, deviceProp.name,
              _ConvertSMVer2ArchName ( deviceProp.major, deviceProp.minor ) );

  cudaDriverGetVersion ( &driverVersion );
  cudaRuntimeGetVersion ( &runtimeVersion );
  printf ( "  CUDA Driver Version %d.%d ",
          driverVersion / 1000,  ( driverVersion % 100 ) / 10 );
  if ( driverVersion / 1000 == 10 )
    printf ( "[2018-]" );
  else if ( driverVersion / 1000 == 11 )
    printf ( "[2020-]" );
  else
    printf ( "[pre 2018]" );
  printf ( " / Runtime Version %d.%d ",
          runtimeVersion / 1000,  ( runtimeVersion % 100 ) / 10 );
  if ( runtimeVersion / 1000 == 8 )
    printf ( "[2016-]\n" );
  else if ( runtimeVersion / 1000 == 9 )
    printf ( "[2017-]\n" );
  else if ( runtimeVersion / 1000 == 10 )
    printf ( "[2018-]\n" );
  else if ( runtimeVersion / 1000 == 11 )
    printf ( "[2020-]\n" );
  else
    printf ( "[pre 2016]\n" );

  printf ( "  CUDA Capability Major/Minor version number:    %d.%d ",
          deviceProp.major, deviceProp.minor );
  switch ( deviceProp.major )
  {
    case 1:
      printf ( "(Tesla [2008-] based)\n" );
      break;
    case 2:
      printf ( "(Fermi [2010-] based)\n" );
      break;
    case 3:
      printf ( "(Kepler [2012-] based)\n" );
      break;
    case 5:
      printf ( "(Maxwell [2014-] based)\n" );
      break;
    case 6:
      printf ( "(Pascal [2016-] based)\n" );
      break;
    case 7:
      if ( deviceProp.minor == 5 )
        printf ( "(Turing [2018-] based)\n" );
      else
        printf ( "(Volta [2017-] based)\n" );
      break;
    case 8:
      printf ( "(Ampere [2021-] based)\n" );
      break;
    default:
      printf ( "(unknown architecture type...)\n" );
      break;
  }

  printf ( "  GPU Clock rate:                                %.0f MHz (%0.2f "
          "GHz)\n", deviceProp.clockRate * 1e-3f,
          deviceProp.clockRate * 1e-6f );
  printf ( "  Memory Clock rate:                             %.0f MHz\n",
          deviceProp.memoryClockRate * 1e-3f );
  printf ( "  Memory Bus Width:                              %d-bit\n",
          deviceProp.memoryBusWidth );

  printf ( "  Total amount of global memory:                 %.2f GBytes (%llu "
          "bytes)\n",  (float) deviceProp.totalGlobalMem / pow ( 1024.0, 3 ),
           (unsigned long long) deviceProp.totalGlobalMem );
  if  ( deviceProp.l2CacheSize )
  {
      printf ( "  L2 Cache Size:                                 %d bytes\n",
              deviceProp.l2CacheSize );
  }

  printf ( "  Total amount of shared memory per block:       %lu bytes\n",
          deviceProp.sharedMemPerBlock );
  printf ( "  Total number of registers available per block: %d\n",
          deviceProp.regsPerBlock );
  printf ( "  Total amount of constant memory:               %lu bytes\n",
          deviceProp.totalConstMem );
  printf ( "  Max Texture Dimension Size (x, y, z)           1D = (%d), "
          "2D = (%d, %d), 3D = (%d, %d, %d)\n", deviceProp.maxTexture1D,
          deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
          deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
          deviceProp.maxTexture3D[2] );
  printf ( "  Max Layered Texture Size (dim) x layers        1D = (%d) x %d, "
          "2D = (%d, %d) x %d\n", deviceProp.maxTexture1DLayered[0],
          deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
          deviceProp.maxTexture2DLayered[1],
          deviceProp.maxTexture2DLayered[2] );
  printf ( "  Max Surface Dimension Size (x, y, z)           1D = (%d), "
          "2D = (%d, %d), 3D = (%d, %d, %d)\n", deviceProp.maxSurface1D,
          deviceProp.maxSurface2D[0], deviceProp.maxSurface2D[1],
          deviceProp.maxSurface3D[0], deviceProp.maxSurface3D[1],
          deviceProp.maxSurface3D[2] );
  printf ( "  Max Layered Surface Size (dim) x layers        1D = (%d) x %d, "
          "2D = (%d, %d) x %d\n", deviceProp.maxSurface1DLayered[0],
          deviceProp.maxSurface1DLayered[1], deviceProp.maxSurface2DLayered[0],
          deviceProp.maxSurface2DLayered[1],
          deviceProp.maxSurface2DLayered[2] );
  printf ( "  Maximum memory pitch:                          %lu bytes\n",
          deviceProp.memPitch );

  printf ( "  (%2d) MultiProcessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
          deviceProp.multiProcessorCount,
          _ConvertSMVer2Cores ( deviceProp.major, deviceProp.minor ),
          _ConvertSMVer2Cores ( deviceProp.major, deviceProp.minor * deviceProp.multiProcessorCount )
         );
  printf ( "  Maximum number of blocks per multiprocessor:   %d\n",
          deviceProp.maxBlocksPerMultiProcessor );
  printf ( "  Warp size:                                     %d\n",
          deviceProp.warpSize );
  printf ( "  Maximum number of threads per multiprocessor:  %d\n",
          deviceProp.maxThreadsPerMultiProcessor );
  printf ( "  Maximum number of threads per block:           %d\n",
          deviceProp.maxThreadsPerBlock );
  printf ( "  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
          deviceProp.maxThreadsDim[0],
          deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2] );
  printf ( "  Maximum sizes of each dimension of a grid:     %d x %d x %d\n\n",
          deviceProp.maxGridSize[0],
          deviceProp.maxGridSize[1],
          deviceProp.maxGridSize[2] );

  exit ( EXIT_SUCCESS );
}
