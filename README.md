# [Pyramidal Lucas-Kanade](https://robots.stanford.edu/cs223b04/algo_tracking.pdf) Feature Tracker with CUDA Acceleration

This repository aims to replicate the functionality of the Pyramidal Lucas-Kanade optical flow algorithm using CUDA for acceleration. 

## Setup

This repository makes use of Docker for setup and development. To get started, ensure you have Docker installed on your machine. 

Additionally, to use your system's GPU, you will need to install the NVIDIA Container Toolkit. Follow the instructions on the [NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to set it up.

You must ensure your GPU driver is compatible with the CUDA version used in the container, which is 12.8.0. This means your GPU driver must be >=525. Run "nvidia-smi" on your base system to determine your GPU driver version.

Then, you can build and run the Docker container using the provided `Dockerfile`. Additionally, if using Visual Studio Code, you can leverage the Dev Containers feature to automatically set up your development environment.

## Building

To build the project, simply run 

```
./build.sh --arch [x86_64|aarch64]
```