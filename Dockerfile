# EMP Project Dockerfile - Replicating conda environment
# Equivalent to: conda create -n emp python=3.9 && conda activate emp (upgraded for compatibility)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables (equivalent to conda activate)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 \
    PYTHON_VERSION=3.9

# Install system dependencies and Python 3.9 (upgraded for av2 compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    protobuf-compiler \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && python${PYTHON_VERSION} -m pip install --upgrade pip \
    && pip install "pip<24.1" \
    && rm -rf /var/lib/apt/lists/*

# Set working directory (equivalent to being inside the conda environment)
WORKDIR /app

# Copy requirements from the emp directory
COPY ./emp/requirements.txt ./requirements.txt

# Step 1: Install PyTorch 2.1.1 with CUDA 12.1 support (equivalent to your pip command)
# Install NumPy 1.23.x which still has np.bool compatibility
RUN pip install --no-cache-dir "numpy==1.23.5"

# Now install PyTorch with the compatible NumPy already in place
# Use --force-reinstall to avoid corrupted cache issues
RUN pip install --no-cache-dir --force-reinstall --no-deps torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --force-reinstall --no-deps torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --force-reinstall --no-deps torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch dependencies separately
RUN pip install --no-cache-dir pillow requests

# Step 2: Install Dependencies with compatible versions (equivalent to: pip install -r ./requirements.txt)
# Pin NumPy 1.23.5 throughout all installations to ensure av2 compatibility
ENV PIP_CONSTRAINT=/tmp/constraints.txt
RUN echo "numpy==1.23.5" > /tmp/constraints.txt

# Install basic packages first with older protobuf for wandb compatibility
RUN pip install --no-cache-dir \
    tqdm \
    rich==12.4.4 \
    plotly \
    setuptools==59.5.0 \
    protobuf==3.19.0

# Install Hydra separately (lightweight)  
RUN pip install --no-cache-dir hydra-core==1.2.0

# Install ML packages that might be sensitive
RUN pip install --no-cache-dir \
    scikit-learn \
    "matplotlib==3.4.3"

# Install PyTorch ecosystem packages
RUN pip install --no-cache-dir \
    timm \
    pytorch-lightning==1.8.2 \
    torchmetrics==0.10.2

# Install wandb with the exact protobuf version from requirements.txt
RUN pip install --no-cache-dir wandb==0.13.5

# Install Ray last (often problematic)
RUN pip install --no-cache-dir ray

# Verify NumPy version after all installations
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Install ONNX with compatible versions for Python 3.9
RUN pip install --no-cache-dir onnx>=1.12.0

# Install remaining ONNX-related packages (now compatible with Python 3.9)
RUN pip install --no-cache-dir \
    onnxscript \
    onnxruntime

# Step 3: Install av2 (equivalent to: pip install av2)
RUN pip install --no-cache-dir av2

# Verify installations (equivalent to testing in conda environment)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')" \
    && python -c "import av2; print('av2 installed successfully')" \
    && echo "Note: CUDA availability test may show False during build - this is normal without GPU runtime"

# Default command (equivalent to having the conda environment ready to use)
CMD ["bash"]