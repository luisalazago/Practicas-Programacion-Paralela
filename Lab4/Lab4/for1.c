/*******************************************************************************
*                                                                              *
*  for1.c - A program to illustrate using
*           the parallel for directive
*                                                                              *
*   Departamento de Electronica y Ciencias de la Computacion                   *
*   Pontificia Universidad Javeriana - CALI                                    *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <omp.h>

#define SIZE  1024

void vectAdd ( double *a, double *b, double *y, int n );

int main ( int argc, char *argv[] )
{
  double  a[SIZE],
          b[SIZE],
          y[SIZE];

  vectAdd ( a, b, y, SIZE );

  return ( 0 );
}

void vectAdd ( double *a, double *b, double *y, int n )
{
  int i;

  #pragma omp parallel for
  for ( i = 0; i < n; i++ )
    y[i] = a[i] + b[i];
}
