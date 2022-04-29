#include <stdio.h>
#include <mpi.h>

#define MSGTAG 0
#define MSGSIZE 1

int main(int argc, char* argv[]) {
    int  my_rank, p, partner_rank_a, partner_rank_s, buf;
    MPI_Status  status;
    int recvd_tag, recvd_from, recvd_count;

    buf = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    partner_rank_s = (my_rank + 1) % p;
    if(my_rank == 0) partner_rank_a = p - 1;
    else partner_rank_a = my_rank - 1;

    printf("p is: %d\n", p);

    if(!my_rank) {
        buf = my_rank;
        MPI_Send(&buf, MSGSIZE, MPI_INT, partner_rank_s, MSGTAG, MPI_COMM_WORLD);
        MPI_Recv(&buf, MSGSIZE, MPI_INT, partner_rank_a, MSGTAG, MPI_COMM_WORLD, &status);
        recvd_tag = status.MPI_TAG;
        recvd_from = status.MPI_SOURCE;
        MPI_Get_count (&status, MPI_INT, &recvd_count);
        printf("The final value is: %d\n", buf);
        printf("What we expect is: %d\n", (p * (p - 1)) / 2);
    }
    else {
        MPI_Recv(&buf, MSGSIZE, MPI_INT, partner_rank_a, MSGTAG, MPI_COMM_WORLD, &status);
        recvd_tag = status.MPI_TAG;
        recvd_from = status.MPI_SOURCE;
        MPI_Get_count (&status, MPI_INT, &recvd_count);
        printf("Recived count: %d\n", recvd_count);
        buf += my_rank;
        MPI_Send(&buf, MSGSIZE, MPI_INT, partner_rank_s, MSGTAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}