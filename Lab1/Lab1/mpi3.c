/*******************************************************************************
*                                                                              *
*  mpi3.c - A program that illustrates using:                                  *
*             - MPI_Abort, which is used whenever an erroneous condition       *
*               arises. Typically, it terminates all the processes in the      *
*               given communicator                                             *
*             - the MPI_Status structure                                       *
*             - MPI_Get_count, which assists in determining the amount of      *
*               received elements of a given type                              *
*                                                                              *
*   Departamento de Electronica y Ciencias de la Computacion                   *
*   Pontificia Universidad Javeriana - CALI                                    *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define PROCESS0      0

enum role_ranks
{
  SENDER,
  RECEIVER
};

#define MSGTAG        0
#define MSGSIZE       1

#define ERROR_CODE    -1

int main ( int argc, char *argv[] )
{
  int         my_rank,
              p,
              partner_rank,
              buf;
  MPI_Status  status;
  int         recvd_tag,
              recvd_from,
              recvd_count;

  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &p );

  if ( p != 2 )
  {
    if ( my_rank == PROCESS0 )
      printf ( "\n%s is meant to be executed by 2 processes.\n\n", argv[0] );
    /* Terminate (in a non-gracefully manner) the runtime environment */
    MPI_Abort ( MPI_COMM_WORLD, ERROR_CODE );
  }

  partner_rank = my_rank ^ 1;

  /* Process 0 sends and Process 1 receives */

  switch ( my_rank )
  {
    case SENDER:
    {
      buf = 123456;
      MPI_Send ( &buf, MSGSIZE, MPI_INT, partner_rank, MSGTAG, MPI_COMM_WORLD );
      printf ( "proc%d sent %d\n", my_rank, buf );
      break;
    }
    case RECEIVER:
    {
      MPI_Recv ( &buf, MSGSIZE, MPI_INT, partner_rank, MSGTAG, MPI_COMM_WORLD, &status );
      recvd_tag = status.MPI_TAG;
      recvd_from = status.MPI_SOURCE;
      MPI_Get_count ( &status, MPI_INT, &recvd_count );
      printf ( "\tproc%d received %d MPI_INT element(s) from proc%d with tag %d"
                " and contents = %d\n",
                my_rank, recvd_count, recvd_from, recvd_tag, buf );
      break;
    }
  }

  MPI_Finalize ();
  return ( 0 );
}
