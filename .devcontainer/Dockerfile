FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN rm -rf \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
    /usr/lib/firmware \
    /usr/local/cuda/compat/lib

RUN apt-get update && apt-get install -y tzdata && \
    apt-get install -y python3.8 python3.8-dev python3-pip curl git font-manager wget && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install virtualenv

RUN virtualenv -p python3.8 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get install -qq -y \
    htop \
    tmux \
    sudo \
    wget

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD mkdir -p /workspace

ARG USERNAME

# Create the user
RUN groupadd $USERNAME \
    && useradd -g $USERNAME -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ADD .devcontainer/.bashrc /root

ENV SHELL /bin/bash