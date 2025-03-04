FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsqlite3-dev \
    libssl-dev \
    libomp-dev \
    libcurl4-openssl-dev \
    libblas-dev \
    liblapack-dev \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA if needed (commented out by default)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-0 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create build directory
WORKDIR /build

# Clone the repository (or copy from local)
COPY . .

# Create build directory
RUN mkdir -p build

# Configure and build
WORKDIR /build/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DLOCALLLM_USE_CUDA=OFF \
    -DLOCALLLM_USE_OPENMP=ON \
    -DLOCALLLM_USE_BLAS=ON

RUN cmake --build . --config Release -j$(nproc)

# Create runtime image
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    libssl3 \
    libomp5 \
    libcurl4 \
    libblas3 \
    liblapack3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built executable and any necessary files
WORKDIR /app
COPY --from=builder /build/build/bin/localllm /app/
COPY --from=builder /build/LICENSE /app/

# Create data directory
RUN mkdir -p /app/localllm_data

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/localllm"]
CMD ["--dir", "/app/localllm_data", "--port", "8080"]
