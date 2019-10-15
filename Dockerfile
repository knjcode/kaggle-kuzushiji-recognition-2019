ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm

# Install PyTorch
ARG CUDA
RUN conda install pytorch=1.2 torchvision cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

RUN apt-get install -y vim

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git /maskrcnn-benchmark
WORKDIR /maskrcnn-benchmark
COPY ./maskrcnn_benchmark/engine/trainer.py maskrcnn_benchmark/engine/
COPY ./maskrcnn_benchmark/modeling/matcher.py maskrcnn_benchmark/modeling/
COPY ./maskrcnn_benchmark/modeling/roi_heads/box_head/loss.py maskrcnn_benchmark/modeling/roi_heads/box_head/
COPY ./maskrcnn_benchmark/modeling/rpn/loss.py maskrcnn_benchmark/modeling/rpn/
COPY ./maskrcnn_benchmark/config/defaults.py maskrcnn_benchmark/config/
COPY ./maskrcnn_benchmark/data/transforms/__init__.py maskrcnn_benchmark/data/transforms/
COPY ./maskrcnn_benchmark/data/transforms/build.py maskrcnn_benchmark/data/transforms/
COPY ./maskrcnn_benchmark/data/transforms/transforms.py maskrcnn_benchmark/data/transforms/
COPY ./maskrcnn_benchmark/engine/trainer.py maskrcnn_benchmark/engine/
COPY ./maskrcnn_benchmark/engine/bbox_aug.py maskrcnn_benchmark/engine/
COPY ./maskrcnn_benchmark/config/paths_catalog.py maskrcnn_benchmark/config/
COPY ./maskrcnn_benchmark/data/datasets/__init__.py maskrcnn_benchmark/data/datasets/
COPY ./maskrcnn_benchmark/data/datasets/kuzushiji.py maskrcnn_benchmark/data/datasets/
COPY ./maskrcnn_benchmark/data/datasets/evaluation/__init__.py maskrcnn_benchmark/data/datasets/evaluation/
COPY ./maskrcnn_benchmark/data/datasets/evaluation/kuzushiji/__init__.py maskrcnn_benchmark/data/datasets/evaluation/kuzushiji/
COPY ./maskrcnn_benchmark/data/datasets/evaluation/kuzushiji/kuzushiji_eval.py maskrcnn_benchmark/data/datasets/evaluation/kuzushiji/

RUN cd /maskrcnn-benchmark \
 && python setup.py build develop
RUN pip install pandas

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /work
