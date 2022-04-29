/*******************************************************************************
*                                                                              *
*  mpi1.c - A program that illustrates four MPI basic functions:               *
*             MPI_Init                                                         *
*             MPI_Comm_rank                                                    *
*             MPI_Comm_size                                                    *
*             MPI_Finalize                                                     *
*                                                                              *
*   Departamento de Electronica y Ciencias de la Computacion                   *
*   Pontificia Universidad Javeriana - CALI                                    *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <mpi.h>

int main ( int argc, char *argv[] )
{
  int my_rank,
      p;

  /* Initialise MPI runtime */
  MPI_Init ( &argc, &argv );

  /* Determine process rank */
  MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );
  /* Determine total number of processes */
  MPI_Comm_size ( MPI_COMM_WORLD, &p );

  printf ( "I am %d of %d\n", my_rank, p );

  /* Shutdown MPI runtime */
  MPI_Finalize ();
  return ( 0 );
}
