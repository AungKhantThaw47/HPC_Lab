#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Version 2: Each process reads its own data directly from files (Parallel I/O)

int main(int argc, char** argv) {
    int rank, size;
    int rows, cols;
    double *matA = NULL, *matB = NULL, *matC = NULL;
    double *local_A = NULL, *local_B = NULL, *local_C = NULL;
    int local_rows;
    MPI_Request requests[4];
    MPI_Status statuses[4];
    double start_time, end_time, elapsed_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Start timing
    start_time = MPI_Wtime();

    printf("Process %d of %d starting...\n", rank, size);
    
    // All processes open and read dimensions
    FILE *fileA = fopen("matAlarge.txt", "r");
    FILE *fileB = fopen("matBlarge.txt", "r");
    
    if (fileA == NULL || fileB == NULL) {
        printf("Process %d: Error opening matrix files!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // All processes read dimensions
    fscanf(fileA, "%d %d", &rows, &cols);
    fscanf(fileB, "%d %d", &rows, &cols);
    
    printf("Process %d: Read dimensions %d x %d\n", rank, rows, cols);
    
    // Calculate rows per process
    local_rows = rows / size;
    int remainder = rows % size;
    
    // Adjust for remainder rows
    if (rank < remainder) {
        local_rows++;
    }
    
    // Calculate starting row for this process
    int start_row = rank * (rows / size) + (rank < remainder ? rank : remainder);
    
    printf("Process %d: Reading rows %d to %d (total %d rows)\n", 
           rank, start_row, start_row + local_rows - 1, local_rows);
    
    // Allocate local buffers
    local_A = (double*)malloc(local_rows * cols * sizeof(double));
    local_B = (double*)malloc(local_rows * cols * sizeof(double));
    local_C = (double*)malloc(local_rows * cols * sizeof(double));
    
    // Skip to the starting position for this process
    // Each row has 'cols' numbers, need to skip start_row * cols numbers
    double dummy;
    for (int i = 0; i < start_row * cols; i++) {
        fscanf(fileA, "%lf", &dummy);
        fscanf(fileB, "%lf", &dummy);
    }
    
    // Read this process's portion of data
    for (int i = 0; i < local_rows * cols; i++) {
        fscanf(fileA, "%lf", &local_A[i]);
        fscanf(fileB, "%lf", &local_B[i]);
    }
    
    fclose(fileA);
    fclose(fileB);
    
    printf("Process %d: Finished reading data\n", rank);
    
    // Calculate send counts and displacements for gathering results
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int proc_rows = rows / size;
        if (i < remainder) {
            proc_rows++;
        }
        sendcounts[i] = proc_rows * cols;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    // Root allocates result matrix
    if (rank == 0) {
        matC = (double*)malloc(rows * cols * sizeof(double));
    }
    
    // Perform local matrix addition
    for (int i = 0; i < local_rows * cols; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }
    
    printf("Process %d: Computed addition for %d rows (direct read version)\n", rank, local_rows);
    
    // Gather results using non-blocking communication to root
    if (rank == 0) {
        // Copy root's result
        for (int i = 0; i < sendcounts[0]; i++) {
            matC[i] = local_C[i];
        }
        
        // Receive from other processes using non-blocking receives
        for (int i = 1; i < size; i++) {
            MPI_Irecv(&matC[displs[i]], sendcounts[i], MPI_DOUBLE, 
                     i, 2, MPI_COMM_WORLD, &requests[0]);
            MPI_Wait(&requests[0], &statuses[0]);
        }
    } else {
        // Other processes send their results using non-blocking send
        MPI_Isend(local_C, local_rows * cols, MPI_DOUBLE, 
                 0, 2, MPI_COMM_WORLD, &requests[0]);
        MPI_Wait(&requests[0], &statuses[0]);
    }
    
    // Root process writes the result
    if (rank == 0) {
        FILE *fileOut = fopen("result.txt", "w");
        fprintf(fileOut, "%d %d\n", rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(fileOut, "%.6f ", matC[i * cols + j]);
            }
            fprintf(fileOut, "\n");
        }
        
        fclose(fileOut);
        printf("Result written to result.txt\n");
        
        free(matC);
    }
    
    // Clean up
    free(local_A);
    free(local_B);
    free(local_C);
    free(sendcounts);
    free(displs);
    
    // End timing
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Print timing for each process
    printf("Process %d: Execution time = %.6f seconds\n", rank, elapsed_time);
    
    // Root prints total time
    if (rank == 0) {
        printf("\n=== TOTAL EXECUTION TIME (Direct Read): %.6f seconds ===\n\n", elapsed_time);
    }
    
    MPI_Finalize();
    return 0;
}
