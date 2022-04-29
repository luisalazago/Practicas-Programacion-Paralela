/*******************************************************************************
*                                                                              *
*  mpi4.c - A program that illustrates how to gracefully exit when erroneous   *
*           conditions arise                                                   *
*                                                                              *
*   Departamento de Electronica y Ciencias de la Computacion                   *
*   Pontificia Universidad Javeriana - CALI                                    *
*                                                                              *
*******************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define PROCESS0  0
#define MSGSIZE   1

enum role_ranks
{
  SENDER,
  RECEIVER
};

enum msg_tags
{
  ERR_TAG,
  INT_TAG
};

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
    int err_code = ERROR_CODE;

    if ( my_rank == PROCESS0 )
    {
      printf ( "\n%s is meant to be executed by 2 processes.\n\n", argv[0] );
      for ( int rank = 1; rank < p; rank++ )
        MPI_Send ( &err_code, MSGSIZE, MPI_INT, rank, ERR_TAG, MPI_COMM_WORLD );
    }
    else
      MPI_Recv ( &err_code, MSGSIZE, MPI_INT, PROCESS0, ERR_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

    /* Terminate (gracefully) the runtime environment */
    MPI_Finalize ();
    return ( 0 );
  }

  partner_rank = my_rank ^ 1;

  /* Process 0 sends and Process 1 receives */

  switch ( my_rank )
  {
    case SENDER:
    {
      buf = 123456;
      MPI_Send ( &buf, MSGSIZE, MPI_INT, partner_rank, INT_TAG, MPI_COMM_WORLD );
      printf ( "proc%d sent %d\n", my_rank, buf );
      break;
    }
    case RECEIVER:
    {
      MPI_Recv ( &buf, MSGSIZE, MPI_INT, partner_rank, INT_TAG, MPI_COMM_WORLD, &status );
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
