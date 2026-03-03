#!/bin/bash

GROUP_ID="17"

if [ -z "$1" ]; then
    echo "Usage: ./package.sh [serial|omp|mpi]"
    exit 1
fi

TARGET=$1

case $TARGET in
    serial|omp|mpi)
        FILENAME="g${GROUP_ID}${TARGET}.zip"
        echo "Creating $FILENAME..."
        
        # This zips the directory and its contents
        zip -r "$FILENAME" "$TARGET"
        
        echo "Done!"
        ;;
    *)
        echo "Invalid target: $TARGET. Choose serial, omp, or mpi."
        exit 1
        ;;
esac
