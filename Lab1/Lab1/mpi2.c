/*******************************************************************************
*                                                                              *
*  mpi2.c - A program that illustrates sending/receiving messages in MPI       *
*           using:                                                             *
*              MPI_Send                                                        *
*              MPI_Recv                                                        *
*                                                                              *
*   Departamento de Electronica y Ciencias de la Computacion                   *
*   Pontificia Universidad Javeriana - CALI                                    *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define SENDER 0
#define RECEIVER 1

#define MSGTAG   0
#define MSGSIZE  1

int main ( int argc, char *argv[] )
{
  int         my_rank,
              buf;

  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );

  /* Process 0 sends and Process 1 receives */

  if ( my_rank == SENDER )
  {
    buf = 123456;
    printf ( "proc%d: buf = %d\n", my_rank, buf );
    MPI_Send( &buf, MSGSIZE, MPI_INT, RECEIVER, MSGTAG, MPI_COMM_WORLD );
    printf ( "proc%d sent %d\n", my_rank, buf );
  }
  else
  if ( my_rank == RECEIVER )
  {
    printf ( "\tproc%d: buf = %d\n", my_rank, buf );
    MPI_Recv ( &buf, MSGSIZE, MPI_INT, SENDER, MSGTAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    printf ( "\tproc%d received %d\n", my_rank, buf );
  }

  MPI_Finalize ();
  return ( 0 );
}
