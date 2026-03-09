# Flow-Guided Krylov - Docker Environment
# Supports both CPU and GPU execution

ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Pin numpy <2 to stay compatible with PyTorch 2.2.0 (compiled against numpy 1.x)
RUN pip install --no-cache-dir "numpy<2"

# Install CuPy 13.x (last version supporting numpy 1.x) for GPU acceleration
RUN pip install --no-cache-dir "cupy-cuda12x<14" || echo "CuPy installation skipped (CPU mode)"

# GPU FCI is embedded in src/utils/gpu_fci.py (requires CuPy only, no gpu4pyscf package)

# Install CUDA-Q for quantum circuit simulation (Path A: exp_pauli gates)
RUN pip install --no-cache-dir cuda-quantum-cu12 || echo "CUDA-Q installation skipped"

# Set PYTHONPATH for direct script execution
ENV PYTHONPATH=/app/src
# Disable Python output buffering for real-time Docker logging
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "examples/validate_small_systems.py"]
