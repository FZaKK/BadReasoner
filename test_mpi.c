#include <mpi.h>
#include <stdio.h>
int main() {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Rank %d: MPI works!\n", rank);
    MPI_Finalize();
    return 0;
}
