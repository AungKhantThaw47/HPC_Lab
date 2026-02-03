#!/bin/bash

# Script to compile and run MPI programs
# Usage: ./run.sh <program_name> [num_processes] [host]

# Default values
NUM_PROCS=${2:-4}  # Default 4 processes
HOST=${3:-localhost:$NUM_PROCS}  # Default localhost

# Check if program name is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <program_name> [num_processes] [host]"
    echo "Example: ./run.sh lab1_mpi_hello_world 6"
    echo "Example: ./run.sh lab1_blocking_add 4"
    exit 1
fi

PROGRAM_NAME=$1
SOURCE_FILE="${PROGRAM_NAME}.c"
EXECUTABLE="${PROGRAM_NAME}"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found!"
    exit 1
fi

echo "========================================="
echo "Compiling: $SOURCE_FILE"
echo "========================================="

# Compile
mpicc -o "$EXECUTABLE" "$SOURCE_FILE" -lm

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful!"
echo ""
echo "========================================="
echo "Running: $EXECUTABLE with $NUM_PROCS processes"
echo "========================================="
echo ""

# Run
mpirun -n "$NUM_PROCS" --host "$HOST" "./$EXECUTABLE"

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Execution completed successfully!"
else
    echo "✗ Execution failed with exit code: $EXIT_CODE"
fi
echo "========================================="

exit $EXIT_CODE
