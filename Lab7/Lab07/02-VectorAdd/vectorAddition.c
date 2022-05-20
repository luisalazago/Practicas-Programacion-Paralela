/*******************************************************************************
*                                                                              *
*  vectorAddition.c -                                                          *
*                                                                              *
*   Adds two vectors on the host (CPU)                                         *
*                                                                              *
*                     Departamento de Electronica y Ciencias de la Computacion *
*                     Pontificia Universidad Javeriana - CALI                  *
*                                                                              *
*******************************************************************************/

#include    <stdio.h>
#include    <stdlib.h>

/***( Manifest Constants )************************************************/

#define VEC_LEN   128

/***( Program Code )******************************************************/

/*--( Support functions )------------------------------------------------*/

/* Add two vectors of length VEC_LEN */
void add ( int *augend, int *addend, int *result )
{
  int i;

  for ( i = 0; i < VEC_LEN; i++ )
    result [i] = augend [i] + addend [i];
}

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
  int *augend, *addend, *result;
  int size = VEC_LEN * sizeof (int);

  /* Allocate space for copies of augend, addend, result; setup input values */
  augend = (int *) malloc ( size ); init_vect ( augend, VEC_LEN );
  addend = (int *) malloc ( size ); init_vect ( addend, VEC_LEN );
  result = (int *) malloc ( size );

  /* Call add () */
  add ( augend, addend, result );

  int i;

  for ( i = 0; i < VEC_LEN; i++ )
    printf ( "%d + %d = %d\n", augend [i], addend [i], result [i] );

  /* Cleanup */
  free ( augend );
  free ( addend );
  free ( result );

  return ( 0 );
}
