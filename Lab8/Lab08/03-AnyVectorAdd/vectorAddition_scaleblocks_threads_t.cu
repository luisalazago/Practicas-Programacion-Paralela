/*******************************************************************************
*                                                                              *
*  vectorAddition_scaleblocks_threads.cu -                                     *
*                                                                              *
*   Adds any two vectors on the device (GPU) using several blocks and several  *
*   threads per block                                                          *
*                                                                              *
*                     Departamento de Electronica y Ciencias de la Computacion *
*                     Pontificia Universidad Javeriana - CALI                  *
*                                                                              *
*******************************************************************************/

#include    <stdio.h>
#include    <stdlib.h>

/***( Manifest Constants )************************************************/

#define N                 256 * 1024 * 1024
#define THREADS_PER_BLOCK 512

/***( Code to be executed on the device (GPU) )****************************/

/* Add two vectors */
__global__ void add ( int *augend, int *addend, int *result )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  while ( index < N )
  {
    result [index] = augend [index] + addend [index];
    /* shift by the total number of threads in the grid */
    index += gridDim.x * THREADS_PER_BLOCK;
  }
}

/***( Code to be executed on the host (CPU) )******************************/

/*--( Support functions )------------------------------------------------*/

/* Initialise a vector of the given length */
void init_vect ( int *vector, int length )
{
  int i;

  for ( i = 0; i < length; i++ )
    vector[i] = i;
}

/*--( Main function )----------------------------------------------------*/
int main ( void )
{
  int *augend, *addend, *result;        /* host copies of augend, addend, result */
  int *d_augend, *d_addend, *d_result;  /* device copies of augend, addend, result */
  int size = N * sizeof (int);

  /* Allocate space for host copies of augend, addend, result; setup input values */
  augend = (int *) malloc ( size ); init_vect ( augend, N );
  addend = (int *) malloc ( size ); init_vect ( addend, N );
  result = (int *) malloc ( size );

  /* create event objects timers */
  cudaEvent_t tstart,
              tstop;

  cudaEventCreate ( &tstart );
  cudaEventCreate ( &tstop );

  /* record event timer tstart on stream 0 */
  cudaEventRecord ( tstart, 0 );

  /* Allocate space for device copies of augend, addend, result */
  cudaMalloc ( (void **) &d_augend, size);
  cudaMalloc ( (void **) &d_addend, size);
  cudaMalloc ( (void **) &d_result, size);

  /* Copy inputs to device */
  cudaMemcpy ( d_augend, augend, size, cudaMemcpyHostToDevice );
  cudaMemcpy ( d_addend, addend, size, cudaMemcpyHostToDevice );

  /* Get number of available MPs and warp size */
  int deviceId;
  cudaGetDevice ( &deviceId );

  cudaDeviceProp props;
  cudaGetDeviceProperties ( &props, deviceId );

  int MultiProcs = props.multiProcessorCount;
  int warpSize = props.warpSize;

  /* Launch add () kernel on GPU */
  printf ( "GPU vector addition of %d elements using %d blocks, %d threads per block; warp size = %d\n",
           N, 32 * MultiProcs, THREADS_PER_BLOCK, warpSize );
  add <<< 32 * MultiProcs, THREADS_PER_BLOCK >>> ( d_augend, d_addend, d_result );

  /* Copy result back to host */
  cudaMemcpy ( result, d_result, size, cudaMemcpyDeviceToHost );

  /* record event timer tstop on stream 0 */
  cudaEventRecord ( tstop, 0 );
  /* wait until the completion of all work currently captured in tstop event */
  cudaEventSynchronize ( tstop );

  /* computes the elapsed time between tstop and tstart events; display timing results */
  float   elapsedTime;
  cudaEventElapsedTime ( &elapsedTime, tstart, tstop );
  printf ( "CUDA processing elapsed time: %3.1f ms\n", elapsedTime );

  /* verify that the calculation is correct */
  bool success = true;

  for ( int i = 0; i < N; i++ )
  {
    if ( result[i] != ( 2 * i ) )
    {
      printf ( "Aaaargh! Result at element %d (%d) doesn't match golden ref (%d)!\n",
               i, result[i], 2 * i );
      success = false;
    }
  }
  if ( success )
    printf ( "GPU vector addition of %d elements matches golden ref\n", N );

  /* Cleanup */
  cudaFree ( d_augend );
  cudaFree ( d_addend );
  cudaFree ( d_result );

  free ( augend );
  free ( addend );
  free ( result );

  return ( 0 );
}
