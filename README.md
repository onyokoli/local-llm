# LocalLLM: Local Fine-tuning Infrastructure

LocalLLM is a comprehensive C++ framework for locally fine-tuning and serving Large Language Models (LLMs) on consumer hardware. It enables developers and researchers to customize open-source LLMs with their own data without requiring cloud resources or expensive GPU clusters.

## Features

- **Local Model Fine-tuning**: Fine-tune models directly on your hardware
- **Multiple Optimization Techniques**:
  - LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
  - QLoRA (Quantized Low-Rank Adaptation) for fine-tuning on consumer GPUs
  - Full fine-tuning for maximum customization
- **Efficient Inference**: Serve fine-tuned models with optimized inference
- **Flexible Dataset Handling**: Support for various dataset formats
- **Model Management**: Simple commands for downloading, quantizing, and managing models
- **REST API**: HTTP server for model inference and management
- **Command Line Interface**: Easy-to-use CLI for all operations

## Requirements

- C++17 compatible compiler (GCC 8+, Clang 10+, MSVC 2019+)
- CMake 3.14+
- CUDA Toolkit 11.4+ (optional, for GPU acceleration)
- 16GB+ RAM for small models, 32GB+ recommended
- Modern CPU with AVX2 support
- NVIDIA GPU with 8GB+ VRAM (optional, for faster fine-tuning and inference)

## Dependencies

- [ggml](https://github.com/ggerganov/ggml): Tensor library for efficient computation
- [llama.cpp](https://github.com/ggerganov/llama.cpp): LLaMA model implementation
- [nlohmann/json](https://github.com/nlohmann/json): JSON parsing
- [CLI11](https://github.com/CLIUtils/CLI11): Command-line argument parsing
- [spdlog](https://github.com/gabime/spdlog): Fast C++ logging library
- [sqlite3](https://www.sqlite.org): Database for model and training metadata
- [cpp-httplib](https://github.com/yhirose/cpp-httplib): HTTP server for REST API

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-username/localllm.git
cd localllm

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc)
```

### Using Docker

```bash
# Build the Docker image
docker build -t localllm .

# Run the container
docker run -p 8080:8080 -v /path/to/data:/app/localllm_data localllm
```

## Quick Start

### Basic Usage

```bash
# Initialize LocalLLM with default settings
./localllm

# Download a model
> download-model facebook/opt-1.3b

# Import a dataset
> import-dataset /path/to/dataset.jsonl my-dataset

# Fine-tune the model using LoRA
> finetune facebook/opt-1.3b my-dataset --method lora --lr 5e-5 --epochs 3

# Generate text with the fine-tuned model
> generate facebook/opt-1.3b-ft-20240301_123456 "Write a poem about machine learning:"
```

### Using the REST API

The LocalLLM server exposes a REST API for model management and inference:

```bash
# Start the server (default port: 8080)
./localllm --port 8080

# In another terminal or application:
# List available models
curl http://localhost:8080/api/models

# Load a model
curl -X POST http://localhost:8080/api/models/facebook/opt-1.3b-ft-20240301_123456/load

# Generate text
curl -X POST http://localhost:8080/api/models/facebook/opt-1.3b-ft-20240301_123456/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a poem about machine learning:", "temperature": 0.7, "max_tokens": 100}'
```

## Optimization Techniques

### LoRA (Low-Rank Adaptation)

LoRA significantly reduces the number of trainable parameters by representing weight updates as low-rank decompositions. This enables fine-tuning with limited memory resources:

```bash
> finetune facebook/opt-1.3b my-dataset --method lora --lora-rank 8
```

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA combines model quantization with LoRA to further reduce memory requirements, allowing fine-tuning of larger models on consumer hardware:

```bash
> finetune facebook/opt-1.3b my-dataset --method qlora --quantization-type q4_0
```

### Full Fine-tuning

Traditional full fine-tuning updates all model parameters, providing maximum customization but requiring more memory:

```bash
> finetune facebook/opt-1.3b my-dataset --method full
```

## Model Management

```bash
# List available models
> list-models

# Download a model
> download-model facebook/opt-1.3b

# Download and quantize a model
> download-model facebook/opt-1.3b --quantize q4_0

# Remove a model (metadata only)
> remove-model facebook/opt-1.3b

# Remove a model including files
> remove-model facebook/opt-1.3b --delete
```

## Dataset Management

```bash
# List available datasets
> list-datasets

# Import a dataset
> import-dataset /path/to/dataset.jsonl my-dataset

# Import a dataset with format specification
> import-dataset /path/to/dataset.csv my-dataset csv
```

## Fine-tuning

```bash
# Start fine-tuning with default parameters
> finetune facebook/opt-1.3b my-dataset

# Fine-tune with specific hyperparameters
> finetune facebook/opt-1.3b my-dataset --method lora --lr 3e-5 --epochs 5 --batch-size 4 --lora-rank 16

# Stop the current fine-tuning process
> stop-finetune
```

## Inference

```bash
# Generate text
> generate facebook/opt-1.3b-ft-20240301_123456 "Write a poem about machine learning:"

# Adjust generation parameters
> generate facebook/opt-1.3b-ft-20240301_123456 "Write a poem about machine learning:" --temp 0.8 --max-tokens 200 --top-p 0.92
```

## API Reference

LocalLLM provides a REST API for programmatic access to all functionality:

### Models

- `GET /api/models`: List all available models
- `GET /api/models/:id`: Get detailed information about a model
- `POST /api/models/:id/load`: Load a model for inference
- `POST /api/models/:id/unload`: Unload a model to free resources
- `POST /api/models/:id/generate`: Generate text with a model
- `POST /api/models/:id/generate/stream`: Stream generated text in real-time

### Datasets

- `GET /api/datasets`: List all available datasets
- `GET /api/datasets/:id`: Get detailed information about a dataset
- `POST /api/datasets/import`: Import a new dataset

### Fine-tuning

- `POST /api/finetune`: Start a fine-tuning job
- `GET /api/finetune/status`: Get status of current fine-tuning job
- `POST /api/finetune/stop`: Stop the current fine-tuning job

### Server

- `GET /api/status`: Get server status and resource usage

## Data Formats

### Dataset Formats

LocalLLM supports several dataset formats:

#### JSONL (Recommended)

Each line is a separate JSON object:

```jsonl
{"instruction": "Translate this text to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Write a poem about", "input": "artificial intelligence", "output": "In circuits deep where data streams..."}
```

#### Chat Format

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me about quantum computing"}, {"role": "assistant", "content": "Quantum computing is..."}]}
```

#### Text Format

Plain text files for next-token prediction tasks:

```text
Once upon a time, there was a
```

## Configuration

LocalLLM can be configured via command-line arguments or configuration files:

```bash
# Specify base directory
./localllm --dir /path/to/data

# Specify server port
./localllm --port 8888
```

## Project Structure

```
localllm/
├── src/
│   ├── model_manager.cpp        # Model loading/saving/management
│   ├── dataset_manager.cpp      # Dataset processing
│   ├── fine_tuning_engine.cpp   # Core training implementation
│   ├── inference_server.cpp     # Model serving
│   ├── cli.cpp                  # Command-line interface
│   └── main.cpp                 # Application entry point
├── include/
│   ├── model_manager.h
│   ├── dataset_manager.h
│   ├── fine_tuning_engine.h
│   ├── inference_server.h
│   └── cli.h
├── third_party/                 # Dependencies
├── CMakeLists.txt               # Build configuration
├── Dockerfile                   # Docker configuration
└── README.md                    # This file
```

## Advanced Usage

### GPU Memory Optimization

When fine-tuning on GPUs with limited VRAM:

```bash
# Use QLoRA with 4-bit quantization
> finetune llama2-7b my-dataset --method qlora --quantization-type q4_0

# Control GPU layers
> finetune llama2-7b my-dataset --gpu-layers 20
```

### Multi-GPU Support

For systems with multiple GPUs:

```bash
# Distribute model across GPUs
> finetune llama2-70b my-dataset --multi-gpu
```

### Checkpointing

Configure checkpoint behavior to save progress during fine-tuning:

```bash
> finetune llama2-7b my-dataset --checkpoint-steps 100 --checkpoint-dir /path/to/checkpoints
```

### Custom Templates

Customize prompting templates for dataset processing:

```bash
# Create a template file:
# template.json:
{
  "input_template": "[INST] {{instruction}} [/INST]",
  "output_template": "{{output}}"
}

# Use custom template:
> import-dataset dataset.jsonl my-dataset --template template.json
```

## Performance Considerations

- **Memory Usage**: LoRA/QLoRA require significantly less memory than full fine-tuning
  - Full fine-tuning typically requires 2-3x the model size in memory
  - LoRA can reduce memory requirements by 10-100x

- **Disk Space**: Models and checkpoints can be large
  - 7B parameter model requires ~13GB for FP16 weights
  - Quantized models (4-bit) reduce size by 4x

- **CPU vs GPU**:
  - CPU-only training is possible but slow (10-100x slower than GPU)
  - Consumer GPUs (8GB+) work well with QLoRA for 7B models
  - High-end GPUs (24GB+) can handle LoRA for larger models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The GGML library for efficient tensor operations
- The LLaMA.cpp project for optimized LLM implementations
- The LoRA and QLoRA papers for memory-efficient fine-tuning methods

## Contact

For issues, feature requests, or questions, please open an issue on the GitHub repository.

---

Happy fine-tuning!
`: Generate text with a loaded model
- `POST /api/models/:id/generate
