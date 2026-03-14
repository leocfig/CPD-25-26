#!/bin/bash

# 1. Config
GROUP_ID="17"
ROOT="g${GROUP_ID}"

# 2. Check Input
TARGET=$1
if [[ ! "$TARGET" =~ ^(serial|omp|mpi)$ ]]; then
    echo "Usage: ./package.sh [serial|omp|mpi]"
    exit 1
fi

# 3. Define the zip filename
ZIP_NAME="${ROOT}${TARGET}.zip"

echo "Packaging $TARGET into $ZIP_NAME..."

# 4. Create a temporary staging area
# We create a folder structure: g23/serial/
mkdir -p "$ROOT"

# 5. Copy the target folder into the root folder
# -a preserves permissions and copies recursively
cp -a "$TARGET" "$ROOT/"

# 6. Create the Archive
zip -r "$ZIP_NAME" "$ROOT"

# 7. Clean up the temporary folder
rm -rf "$ROOT"

zip -sf "$ZIP_NAME"