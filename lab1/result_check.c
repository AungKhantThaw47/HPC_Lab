#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-6  // Tolerance for floating point comparison

int main(int argc, char** argv) {
    FILE *result_file, *solution_file;
    int result_rows, result_cols, solution_rows, solution_cols;
    double result_val, solution_val;
    int differences = 0;
    int total_elements = 0;
    double max_diff = 0.0;
    
    // Open files
    result_file = fopen("result.txt", "r");
    solution_file = fopen("solutionsmall.txt", "r");
    
    if (result_file == NULL) {
        printf("Error: Cannot open result.txt\n");
        return 1;
    }
    
    if (solution_file == NULL) {
        printf("Error: Cannot open solutionsmall.txt\n");
        fclose(result_file);
        return 1;
    }
    
    // Read dimensions
    fscanf(result_file, "%d %d", &result_rows, &result_cols);
    fscanf(solution_file, "%d %d", &solution_rows, &solution_cols);
    
    printf("Result dimensions: %d x %d\n", result_rows, result_cols);
    printf("Solution dimensions: %d x %d\n", solution_rows, solution_cols);
    
    // Check if dimensions match
    if (result_rows != solution_rows || result_cols != solution_cols) {
        printf("ERROR: Dimensions do not match!\n");
        fclose(result_file);
        fclose(solution_file);
        return 1;
    }
    
    printf("\nComparing matrices...\n");
    
    // Compare values
    for (int i = 0; i < result_rows; i++) {
        for (int j = 0; j < result_cols; j++) {
            if (fscanf(result_file, "%lf", &result_val) != 1) {
                printf("ERROR: Failed to read from result.txt at position (%d, %d)\n", i, j);
                fclose(result_file);
                fclose(solution_file);
                return 1;
            }
            
            if (fscanf(solution_file, "%lf", &solution_val) != 1) {
                printf("ERROR: Failed to read from solutionsmall.txt at position (%d, %d)\n", i, j);
                fclose(result_file);
                fclose(solution_file);
                return 1;
            }
            
            double diff = fabs(result_val - solution_val);
            
            if (diff > max_diff) {
                max_diff = diff;
            }
            
            if (diff > EPSILON) {
                differences++;
                if (differences <= 10) {  // Show first 10 differences
                    printf("  Difference at [%d][%d]: result=%.6f, solution=%.6f, diff=%.6f\n",
                           i, j, result_val, solution_val, diff);
                }
            }
            
            total_elements++;
        }
    }
    
    // Print summary
    printf("\n========== COMPARISON SUMMARY ==========\n");
    printf("Total elements compared: %d\n", total_elements);
    printf("Elements with differences: %d\n", differences);
    printf("Maximum difference: %.10f\n", max_diff);
    printf("Tolerance (EPSILON): %.10f\n", EPSILON);
    
    if (differences == 0) {
        printf("\n✓ SUCCESS: Results match perfectly!\n");
    } else {
        double error_rate = (double)differences / total_elements * 100.0;
        printf("\n✗ FAILURE: %.2f%% of elements differ\n", error_rate);
    }
    printf("========================================\n");
    
    fclose(result_file);
    fclose(solution_file);
    
    return (differences == 0) ? 0 : 1;
}
