# Configure image
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Configure environment variables
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
# ENV MUJOCO_GL="egl"
ENV MUJOCO_GL="osmesa"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg libosmesa6-dev\
    speech-dispatcher libgeos-dev \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Clone repository and install LeRobot in a single layer
RUN pip install vcstool
COPY . /home/lerobot_mujoco_env
WORKDIR /home
RUN vcs import . < ./lerobot_mujoco_env/repos.repos
WORKDIR /home/lerobot
RUN /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[test, aloha, xarm, pusht]" \
        --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install "mink[examples]"
RUN pip install qpsolvers[quadprog]
RUN pip install transformers num2words accelerate
WORKDIR /home

# Execute in bash shell rather than python
CMD ["/bin/bash"]
