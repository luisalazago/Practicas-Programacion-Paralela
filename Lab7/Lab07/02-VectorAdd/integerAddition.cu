/*******************************************************************************
*                                                                              *
*  integerAddition.cu -                                                        *
*                                                                              *
*   Adds two numbers on the device (GPU)                                       *
*                                                                              *
*                     Departamento de Electronica y Ciencias de la Computacion *
*                     Pontificia Universidad Javeriana - CALI                  *
*                                                                              *
*******************************************************************************/

#include  <stdio.h>

/***( Code to be executed on the device (GPU) )****************************/

/* Add two scalars */
__global__ void add ( int *augend, int *addend, int *result )
{
  *result = *augend + *addend;
}

/***( Code to be executed on the host (CPU) )******************************/

/*--( Main function )----------------------------------------------------*/
int main ( void )
{
  int augend,       /* host copies of augend, addend, result */
      addend,
      result;
  int *d_augend,
      *d_addend,
      *d_result;    /* device copies of augend, addend, result */

  int size = sizeof (int);

  /* Allocate space for device copies of augend, addend, result */
  cudaMalloc ( (void **) &d_augend, size );
  cudaMalloc ( (void **) &d_addend, size );
  cudaMalloc ( (void **) &d_result, size );

  /* Setup input values */
  augend = 2;
  addend = 7;

  /* Copy inputs to device */
  cudaMemcpy ( d_augend, &augend, size, cudaMemcpyHostToDevice );
  cudaMemcpy ( d_addend, &addend, size, cudaMemcpyHostToDevice );

  /* Launch add () kernel on GPU */
  add <<< 1, 1 >>> ( d_augend, d_addend, d_result );

  /* Copy result back to host */
  cudaMemcpy ( &result, d_result, size, cudaMemcpyDeviceToHost );

  printf ( "%d + %d = %d\n", augend, addend, result );

  /* Cleanup */
  cudaFree ( d_augend );
  cudaFree ( d_addend );
  cudaFree ( d_result );

  return ( 0 );
}
