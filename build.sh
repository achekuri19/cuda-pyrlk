#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

usage()
{
    echo "usage: ./build.sh --arch [x86_64|aarch64] [--with-cuda] [--clean]"
}

TARGET_ARCH=""
WITH_CUDA="OFF"
BUILD_CLEAN=0

while [ "$1" != "" ]; do
    case $1 in
        --arch )                shift
                                TARGET_ARCH=$1
                                ;;
        --with-cuda )           WITH_CUDA="ON"
                                ;;
        --clean )               BUILD_CLEAN=1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# TODO: add CLI option for build type
BUILD_TYPE="Debug"

# TODO: add CLI option for CUDA architectures
# CUDA_ARCHS="native"

# Validate input
if [ -z "$TARGET_ARCH" ]; then
    usage
    exit 1
elif [ "$TARGET_ARCH" == "x86_64" ]; then
    TOOLCHAIN_FILE="$SCRIPT_DIR/toolchain/x86_64_linux_gcc13.cmake"
    BUILD_DIR="$SCRIPT_DIR/build/x86_64"
elif [ "$TARGET_ARCH" == "aarch64" ]; then
    TOOLCHAIN_FILE="$SCRIPT_DIR/toolchain/aarch64_linux_gcc13.cmake"
    BUILD_DIR="$SCRIPT_DIR/build/aarch64"
else
    echo "Unsupported architecture: $TARGET_ARCH"
    exit 1
fi


# Start main build commands
mkdir -p "$BUILD_DIR"

if [ "$BUILD_CLEAN" -eq 1 ]; then
    # remove everything from build directory
    rm -rf "$BUILD_DIR"/*
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DWITH_CUDA="$WITH_CUDA"
cmake --build "${BUILD_DIR}"

echo "Build completed at: $BUILD_DIR for $TARGET_ARCH with CUDA $WITH_CUDA"



