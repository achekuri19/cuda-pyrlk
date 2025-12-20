#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

usage()
{
    echo "usage: ./build.sh --arch [x86_64|aarch64]"
}

TARGET_ARCH=""

while [ "$1" != "" ]; do
    case $1 in
        --arch )                shift
                                TARGET_ARCH=$1
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
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR"

echo "Build completed at: $BUILD_DIR for $TARGET_ARCH"



