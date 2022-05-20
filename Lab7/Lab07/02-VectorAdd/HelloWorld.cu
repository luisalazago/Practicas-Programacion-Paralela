/*******************************************************************************
*                                                                              *
*  HelloWorld.cu - A true GPU HelloWorld!                                      *
*                                                                              *
*   Adds two vectors on the device (GPU) to print HelloWorld!                  *
*                                                                              *
*                     Departamento de Electronica y Ciencias de la Computacion *
*                     Pontificia Universidad Javeriana - CALI                  *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <stdint.h>

/***( Code to be executed on the device (GPU) )****************************/

/* Add two vectors */
__global__ void HelloWorld ( int8_t *v1, int8_t *v2 )
{
  v1[threadIdx.x] += v2[threadIdx.x];
}

/***( Code to be executed on the host (CPU) )******************************/
int main ( void )
{
  int8_t v1[] = { 36, 51, 69, 40, 70, 8, 15, 32, 24, 86, 75, 55, 12,  3, -4,   8 };
  int8_t v2[] = { 36, 50, 39, 68, 41, 12, 5, 55, 87, 28, 33, 45, 11, 30,  4, -81 };

  int8_t *d_v1;
  int8_t *d_v2;

  int N = 16;
  int size = N * sizeof (int8_t);

  cudaMalloc ( (void **) &d_v1, size );
  cudaMalloc ( (void **) &d_v2, size );

  cudaMemcpy ( d_v1, v1, size, cudaMemcpyHostToDevice );
  cudaMemcpy ( d_v2, v2, size, cudaMemcpyHostToDevice );

  HelloWorld <<< 1, 16 >>> ( d_v1, d_v2 );

  cudaMemcpy ( v1, d_v1, size, cudaMemcpyDeviceToHost );

  cudaFree ( d_v1 );
  cudaFree ( d_v2 );

  printf ( "%s\n", v1 );

  return ( 0 );
}
