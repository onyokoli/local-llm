#ifndef LOCALLLM_TYPES_H
#define LOCALLLM_TYPES_H

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <cstdint>

#include "nlohmann/json.hpp"

// Include llama.cpp headers
#include "llama.h"  // This defines llama_token, llama_model, llama_context

// For convenience
namespace fs = std::filesystem;
using json = nlohmann::json;

// If llama.h is not properly included, we need these fallback declarations
#ifndef LLAMA_H
struct llama_model;
struct llama_context;
typedef int llama_token;  // Define llama_token as int if not defined by llama.h
#endif

namespace localllm {

// ======== Constants ========

constexpr int DEFAULT_CONTEXT_LENGTH = 2048;
constexpr float DEFAULT_LEARNING_RATE = 2e-5f;
constexpr int DEFAULT_BATCH_SIZE = 1;
constexpr int DEFAULT_EPOCHS = 3;
constexpr int DEFAULT_LORA_RANK = 8;
constexpr float DEFAULT_LORA_ALPHA = 16.0f;
constexpr float DEFAULT_WEIGHT_DECAY = 0.01f;
constexpr float DEFAULT_ADAM_BETA1 = 0.9f;
constexpr float DEFAULT_ADAM_BETA2 = 0.999f;
constexpr float DEFAULT_ADAM_EPSILON = 1e-8f;

// ======== Data Structures ========

// Model metadata
struct ModelInfo {
    std::string id;
    std::string name;
    std::string base_path;
    std::string architecture_type;  // "llama", "mistral", etc.
    uint64_t parameter_count;
    int context_length;
    bool is_quantized;
    std::string quantization_method;  // "q4_0", "q4_1", "q5_0", etc.
    std::optional<std::string> parent_model_id;  // For fine-tuned models

    // Fine-tuning specific info (if applicable)
    bool is_finetuned = false;
    std::string finetuning_method;  // "lora", "qlora", "full"
    int lora_rank = 0;
    std::string training_dataset;
    std::string training_date;
    int training_steps = 0;
    float final_loss = 0.0f;

    // Serialize model info to JSON
    json to_json() const;

    // Deserialize from JSON
    static ModelInfo from_json(const json& j);
};

// Dataset information
struct DatasetInfo {
    std::string id;
    std::string name;
    fs::path data_path;
    uint64_t num_examples;
    uint64_t total_tokens;
    std::string format_type; // "jsonl", "text", "csv", etc.
    std::string created_date;
    bool is_processed = false;

    // Format-specific configurations
    json format_config;

    // Training split information
    double val_split_ratio = 0.1;
    uint64_t train_examples = 0;
    uint64_t val_examples = 0;

    // Example shape information
    std::string input_template;
    std::string output_template;
    uint64_t avg_input_length = 0;
    uint64_t avg_output_length = 0;
    uint64_t max_length = 0;

    // Serialize to JSON
    json to_json() const;

    // Deserialize from JSON
    static DatasetInfo from_json(const json& j);
};

// Fine-tuning configuration
struct FineTuneConfig {
    // Training hyperparameters
    float learning_rate = DEFAULT_LEARNING_RATE;
    int batch_size = DEFAULT_BATCH_SIZE;
    int epochs = DEFAULT_EPOCHS;
    float weight_decay = DEFAULT_WEIGHT_DECAY;

    // Adam optimizer parameters
    float adam_beta1 = DEFAULT_ADAM_BETA1;
    float adam_beta2 = DEFAULT_ADAM_BETA2;
    float adam_epsilon = DEFAULT_ADAM_EPSILON;

    // Optimization method
    enum class Method {
        FULL_FINETUNE,
        LORA,
        QLORA
    } method = Method::LORA;

    // LoRA parameters
    int lora_rank = DEFAULT_LORA_RANK;
    float lora_alpha = DEFAULT_LORA_ALPHA;
    float lora_dropout = 0.05f;
    std::vector<std::string> lora_target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};

    // Quantization parameters (for QLoRA)
    std::string quantization_type = "q4_0";  // GGML quantization type

    // Resource constraints
    size_t max_memory_usage = 0;  // 0 means auto-detect
    int num_threads = 0;         // 0 means use all available
    bool use_gpu = true;

    // Checkpointing
    bool enable_checkpointing = true;
    int checkpoint_steps = 100;
    std::string checkpoint_dir = "";

    // Training process
    int gradient_accumulation_steps = 1;
    int warmup_steps = 100;
    float max_grad_norm = 1.0f;

    // Context window
    int context_length = DEFAULT_CONTEXT_LENGTH;

    // Evaluation
    bool eval_during_training = true;
    int eval_steps = 50;

    // Logging
    int logging_steps = 10;

    // Serialize to JSON
    json to_json() const;

    // Deserialize from JSON
    static FineTuneConfig from_json(const json& j);
};

// Inference configuration
struct InferenceConfig {
    float temperature = 0.8f;
    int max_tokens = 512;
    float top_p = 0.95f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    int n_batch = 512;  // Number of tokens to process in parallel
    int context_length = DEFAULT_CONTEXT_LENGTH;
    int num_threads = 0;  // 0 means use all available
    bool use_gpu = true;

    // Streaming settings
    bool stream = false;

    // Sampling seed
    uint64_t seed = 0; // 0 means random

    // Serialize to JSON
    json to_json() const;

    // Deserialize from JSON
    static InferenceConfig from_json(const json& j);
};

// Training example
struct TrainingExample {
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> output_tokens;

    size_t total_tokens() const {
        return input_tokens.size() + output_tokens.size();
    }
};

// Generation result
struct GenerationResult {
    bool success;
    std::string text;
    std::string error_message;
    int token_count;
    int64_t generation_time_ms;
};

} // namespace localllm

#endif // LOCALLLM_TYPES_H
