FROM ubuntu:latest

USER root

# Set the environment variable
ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /workspace

# Copy the poetry.lock and pyproject.toml
COPY pyproject.toml poetry.lock ./

# Install needed packages and python
RUN apt-get -y update && \
    apt-get -y --fix-missing install git curl && \
    apt-get -y --fix-missing install python3 python3-pip python3-venv && \
    apt-get -y --fix-missing install ffmpeg
RUN echo 'alias python=python3' >> ~/.bashrc && \
    echo 'alias pip=pip3' >> ~/.bashrc && \
    . ~/.bashrc

# Install Pyenv
RUN git clone https://github.com/pyenv/pyenv ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PATH:$PYENV_ROOT/bin"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc && \
    . ~/.bashrc

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    . ~/.bashrc && \
    poetry install
