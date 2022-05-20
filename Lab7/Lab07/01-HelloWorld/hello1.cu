/*******************************************************************************
*                                                                              *
*  hello1.cu -                                                                 *
*                                                                              *
*   Print a message from the host (CPU)                                        *
*   Print a message from the device (GPU) using five threads                   *
*                                                                              *
*                     Departamento de Electronica y Ciencias de la Computacion *
*                     Pontificia Universidad Javeriana - CALI                  *
*                                                                              *
*******************************************************************************/

#include <stdio.h>

/***( Code to be executed on the device (GPU) )****************************/

__global__ void GPUkernel ( void )
{
  printf ( "Hello World from block %d, thread %d running on GPU!\n",
           blockIdx.x, threadIdx.x );
}

/***( Code to be executed on the host (CPU) )******************************/

void CPUfunction ( void )
{
  printf ( "Hello World from CPU!\n" );
}

int main ( void )
{
  CPUfunction ();
  GPUkernel <<< 1, 5 >>> ();
  cudaDeviceSynchronize ();

  return ( 0 );
}
