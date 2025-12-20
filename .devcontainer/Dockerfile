FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    sudo \
    bash-completion \
    wget \
    cmake \
    # TODO: Make Dockerfile architecture-aware to install cross-compilation tools based on host platform
    # (install aarch64 tools on x86_64 hosts, install x86_64 tools on aarch64 hosts)
    gcc-13-aarch64-linux-gnu \
    g++-13-aarch64-linux-gnu \
    qemu-user-static