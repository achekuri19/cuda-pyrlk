# [Pyramidal Lucas-Kanade](https://robots.stanford.edu/cs223b04/algo_tracking.pdf) Feature Tracker with CUDA Acceleration

This repository aims to replicate the functionality of the Pyramidal Lucas-Kanade optical flow algorithm using CUDA for acceleration. 

## Setup

This repository makes use of [Docker](https://docs.docker.com/engine/install/) for setup and development. To get started, ensure you have Docker installed on your machine. 

Additionally, to use your system's GPU, you will need to install the NVIDIA Container Toolkit. Follow the instructions on the [NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to set it up.

You must ensure your GPU driver is compatible with the CUDA version used in the container, which is 12.8.0. This means your GPU driver must be >=525. Run "nvidia-smi" on your base system to determine your GPU driver version.

### Option 1: Using VS Code Dev Containers

If using Visual Studio Code, you can leverage the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) feature to automatically set up your development environment with the container.

### Option 2: Using Docker CLI

To build and run the Docker container manually via CLI:

1. **Build the Docker image:**
   ```bash
   docker build -t cuda-pyrlk-dev .devcontainer/
   ```

2. **Run the Docker container:**
   ```bash
   docker run -it --rm \
     --gpus all \
     --user ubuntu \
     -v "$(pwd):/workspaces/cuda-pyrlk" \
     -v "$HOME/.ssh:/home/ubuntu/.ssh:ro" \
     -w /workspaces/cuda-pyrlk \
     cuda-pyrlk-dev
   ```

   This command:
   - `--gpus all`: Enables GPU access in the container
   - `--user ubuntu`: Runs as the ubuntu user (matching Dev Containers behavior)
   - `-v "$(pwd):/workspaces/cuda-pyrlk"`: Mounts the repository at `/workspaces/cuda-pyrlk`
   - `-v "$HOME/.ssh:/home/ubuntu/.ssh:ro"`: Mounts your SSH credentials (read-only)
   - `-w /workspaces/cuda-pyrlk`: Sets the working directory
   - `--rm`: Automatically removes the container when it exits
   - `-it`: Runs interactively with a terminal

   **Note:** If you need to persist the container state between sessions, remove the `--rm` flag and use `docker start` to restart the container.

### Option 3: Build Natively

If you prefer to build the project natively on your host machine, you can do so by installing the necessary dependencies and following the build instructions.

## Building

To build the project, simply run 

```
./build.sh --arch [x86_64|aarch64]
```

## Running
TODO: Add actual binary. To run, use the following command

```
./build/<arch>/hello
```

NOTE: If cross compiling, you will have to make use of QEMU to run your built binary. It is installed by default in the Docker container. For example, if your host machine is x86_64 and you are targeting aarch64, you would run:

```
qemu-aarch64-static ./build/aarch64/hello
```