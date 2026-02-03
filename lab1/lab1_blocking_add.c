#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int rows, cols;
    double *matA = NULL, *matB = NULL, *matC = NULL;
    double *local_A = NULL, *local_B = NULL, *local_C = NULL;
    int local_rows;
    double start_time, end_time, elapsed_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Start timing
    start_time = MPI_Wtime();
    
    // Root process reads the matrices
    if (rank == 0) {
        FILE *fileA = fopen("matAlarge.txt", "r");
        FILE *fileB = fopen("matBlarge.txt", "r");
        
        if (fileA == NULL || fileB == NULL) {
            printf("Error opening matrix files!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read dimensions
        fscanf(fileA, "%d %d", &rows, &cols);
        fscanf(fileB, "%d %d", &rows, &cols);
        
        // Allocate memory for full matrices
        matA = (double*)malloc(rows * cols * sizeof(double));
        matB = (double*)malloc(rows * cols * sizeof(double));
        matC = (double*)malloc(rows * cols * sizeof(double));
        
        // Read matrix A
        for (int i = 0; i < rows * cols; i++) {
            fscanf(fileA, "%lf", &matA[i]);
        }
        
        // Read matrix B
        for (int i = 0; i < rows * cols; i++) {
            fscanf(fileB, "%lf", &matB[i]);
        }
        
        fclose(fileA);
        fclose(fileB);
        
        printf("Matrices loaded: %d x %d\n", rows, cols);
    }
    
    // Broadcast dimensions to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate rows per process
    local_rows = rows / size;
    int remainder = rows % size;
    
    // Adjust for remainder rows
    if (rank < remainder) {
        local_rows++;
    }
    
    // Allocate local buffers
    local_A = (double*)malloc(local_rows * cols * sizeof(double));
    local_B = (double*)malloc(local_rows * cols * sizeof(double));
    local_C = (double*)malloc(local_rows * cols * sizeof(double));
    
    // Create arrays for send counts and displacements
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
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
    }
    
    // Scatter matrix A rows to all processes
    MPI_Scatterv(matA, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Scatter matrix B rows to all processes
    MPI_Scatterv(matB, sendcounts, displs, MPI_DOUBLE,
                 local_B, local_rows * cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Perform local matrix addition
    for (int i = 0; i < local_rows * cols; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }
    
    printf("Process %d: Computed addition for %d rows\n", rank, local_rows);
    
    // Gather results back to root
    MPI_Gatherv(local_C, local_rows * cols, MPI_DOUBLE,
                matC, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Root process writes the result
    if (rank == 0) {
        FILE *fileOut = fopen("result_add.txt", "w");
        fprintf(fileOut, "%d %d\n", rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(fileOut, "%.6f ", matC[i * cols + j]);
            }
            fprintf(fileOut, "\n");
        }
        
        fclose(fileOut);
        printf("Result written to result_add.txt\n");
        
        free(matA);
        free(matB);
        free(matC);
        free(sendcounts);
        free(displs);
    }
    
    // Clean up
    free(local_A);
    free(local_B);
    free(local_C);
    
    // End timing
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Print timing for each process
    printf("Process %d: Execution time = %.6f seconds\n", rank, elapsed_time);
    
    // Root prints total time
    if (rank == 0) {
        printf("\n=== TOTAL EXECUTION TIME (Blocking): %.6f seconds ===\n\n", elapsed_time);
    }
    
    MPI_Finalize();
    return 0;
}
