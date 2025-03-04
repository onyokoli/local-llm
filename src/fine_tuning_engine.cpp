#include "fine_tuning_engine.h"
#include "model_manager.h"
#include "dataset_manager.h"
#include "utils.h"

#include <ggml.h>
#include <llama.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace localllm {

// LLAMA_API binding for LoRA adapter training
extern "C" {
    LLAMA_API int llama_model_train(
        struct llama_model * model,
        struct llama_context * ctx,
        const float * data,
        int n_tokens,
        int batch_size,
        float learning_rate,
        bool use_lora,
        int lora_rank);

    LLAMA_API int llama_save_adapter(
        struct llama_context * ctx,
        const char * path);

    LLAMA_API int llama_load_adapter(
        struct llama_context * ctx,
        const char * path);
}

FineTuningEngine::FineTuningEngine(ModelManager& model_manager, DatasetManager& dataset_manager)
    : model_manager_(model_manager), dataset_manager_(dataset_manager) {

    // Create output directory for logs if it doesn't exist
    fs::path log_dir = model_manager_.getBaseDir() / "logs" / "training";
    if (!fs::exists(log_dir)) {
        fs::create_directories(log_dir);
    }
}

FineTuningEngine::~FineTuningEngine() {
    stopTraining(true); // Wait for training to complete
    if (training_thread_.joinable()) {
        training_thread_.join();
    }
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

bool FineTuningEngine::startFineTuning(
    const std::string& model_id,
    const std::string& dataset_id,
    const FineTuneConfig& config) {

    std::lock_guard<std::mutex> lock(training_mutex_);

    if (is_training_.load()) {
        spdlog::error("Cannot start fine-tuning: another training session is in progress");
        return false;
    }

    // Get model and dataset info
    ModelInfo model_info;
    if (!model_manager_.getModelInfo(model_id, model_info)) {
        spdlog::error("Cannot start fine-tuning: model '{}' not found", model_id);
        return false;
    }

    DatasetInfo dataset_info;
    if (!dataset_manager_.getDatasetInfo(dataset_id, dataset_info)) {
        spdlog::error("Cannot start fine-tuning: dataset '{}' not found", dataset_id);
        return false;
    }

    // Setup the log file
    std::string timestamp = getCurrentTimestamp(true);
    log_file_path_ = (model_manager_.getBaseDir() / "logs" / "training" /
                      (model_id + "_" + dataset_id + "_" + timestamp + ".log")).string();

    log_file_.open(log_file_path_, std::ios::out | std::ios::trunc);
    if (!log_file_.is_open()) {
        spdlog::error("Cannot start fine-tuning: failed to open log file at '{}'", log_file_path_);
        return false;
    }

    // Reset state
    state_ = TrainingState();
    state_.model_id = model_id;
    state_.dataset_id = dataset_id;
    state_.config = config;
    state_.start_time = std::chrono::steady_clock::now();

    stop_requested_ = false;
    is_training_ = true;

    // Start training in a separate thread
    training_thread_ = std::thread([this, model_info, dataset_info, config]() {
        try {
            runTrainingLoop(model_info, dataset_info, config);
        } catch (const std::exception& e) {
            std::string error_msg = "Training failed with exception: " + std::string(e.what());
            logTrainingError(error_msg);
            spdlog::error(error_msg);
        } catch (...) {
            std::string error_msg = "Training failed with unknown exception";
            logTrainingError(error_msg);
            spdlog::error(error_msg);
        }

        // Mark training as completed
        {
            std::lock_guard<std::mutex> lock(training_mutex_);
            is_training_ = false;
            state_.end_time = std::chrono::steady_clock::now();
        }

        training_cv_.notify_all();

        if (log_file_.is_open()) {
            log_file_.close();
        }
    });

    return true;
}

void FineTuningEngine::stopTraining(bool wait_for_completion) {
    stop_requested_ = true;

    if (wait_for_completion) {
        std::unique_lock<std::mutex> lock(training_mutex_);
        training_cv_.wait(lock, [this] { return !is_training_.load(); });
    }
}

bool FineTuningEngine::isTraining() const {
    return is_training_.load();
}

FineTuningEngine::TrainingState FineTuningEngine::getTrainingState() const {
    std::lock_guard<std::mutex> lock(training_mutex_);
    return state_;
}

std::string FineTuningEngine::getLogFilePath() const {
    return log_file_path_;
}

void FineTuningEngine::runTrainingLoop(
    const ModelInfo& model_info,
    const DatasetInfo& dataset_info,
    const FineTuneConfig& config) {

    spdlog::info("Starting fine-tuning of model '{}' with dataset '{}'",
                 model_info.id, dataset_info.id);

    logTrainingConfig(model_info, dataset_info, config);

    // Prepare paths
    fs::path model_path = model_manager_.getModelPath(model_info.id);
    fs::path dataset_path = dataset_manager_.getDatasetPath(dataset_info.id);
    fs::path checkpoint_dir = model_manager_.getBaseDir() / "checkpoints" /
                             (model_info.id + "_" + dataset_info.id + "_" + getCurrentTimestamp(true));

    if (config.save_checkpoints && !fs::exists(checkpoint_dir)) {
        fs::create_directories(checkpoint_dir);
    }

    // Create LLAMA context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_length;
    ctx_params.seed = config.seed;

    // Load the model
    spdlog::info("Loading model from '{}'", model_path.string());

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;

    try {
        // Set up model params for fine-tuning
        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = false;

        // Apply quantization if using QLoRA
        if (config.method == FineTuneMethod::QLORA) {
            model_params.use_mmap = false;
            model_params.use_mlock = true;

            // Set quantization parameters based on config
            switch (config.quantization_type) {
                case QuantizationType::Q4_0:
                    model_params.type = LLAMA_FTYPE_MOSTLY_Q4_0;
                    break;
                case QuantizationType::Q4_1:
                    model_params.type = LLAMA_FTYPE_MOSTLY_Q4_1;
                    break;
                case QuantizationType::Q5_0:
                    model_params.type = LLAMA_FTYPE_MOSTLY_Q5_0;
                    break;
                case QuantizationType::Q5_1:
                    model_params.type = LLAMA_FTYPE_MOSTLY_Q5_1;
                    break;
                case QuantizationType::Q8_0:
                    model_params.type = LLAMA_FTYPE_MOSTLY_Q8_0;
                    break;
                default:
                    model_params.type = LLAMA_FTYPE_ALL_F16;
                    break;
            }
        } else {
            // For LoRA or full fine-tuning, use F16 weights
            model_params.type = LLAMA_FTYPE_ALL_F16;
        }

        // Load the model with fine-tuning parameters
        model = llama_load_model_from_file(model_path.string().c_str(), model_params);
        if (!model) {
            throw std::runtime_error("Failed to load model");
        }

        // Create context for training
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }

        // Load data
        spdlog::info("Loading training data from '{}'", dataset_path.string());

        // Split dataset into train and validation sets
        auto [train_examples, val_examples] = dataset_manager_.loadAndSplitDataset(
            dataset_info.id, config.validation_split);

        // Save validation set for future evaluation
        fs::path val_path = checkpoint_dir / "validation.bin";
        if (config.save_checkpoints && !val_examples.empty()) {
            // Save validation examples to a binary file
            std::ofstream val_file(val_path, std::ios::binary);
            if (val_file.is_open()) {
                size_t num_examples = val_examples.size();
                val_file.write(reinterpret_cast<const char*>(&num_examples), sizeof(size_t));

                for (const auto& example : val_examples) {
                    size_t len = example.tokens.size();
                    val_file.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
                    val_file.write(reinterpret_cast<const char*>(example.tokens.data()), len * sizeof(llama_token));
                }
                val_file.close();
            }
        }

        // Calculate total steps
        size_t num_batches = (train_examples.size() + config.batch_size - 1) / config.batch_size;
        state_.total_steps = num_batches * config.num_epochs;

        spdlog::info("Starting training with {} training examples, {} validation examples, {} batches, {} epochs",
                    train_examples.size(), val_examples.size(), num_batches, config.num_epochs);

        // Setup LoRA params if using LoRA/QLoRA
        bool use_lora = (config.method == FineTuneMethod::LORA || config.method == FineTuneMethod::QLORA);

        // Create a random number generator for shuffling
        std::mt19937 rng(config.seed);

        // Main training loop
        for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
            if (stop_requested_) {
                spdlog::info("Training stopped by user request");
                break;
            }

            state_.current_epoch = epoch;

            // Shuffle training data at the start of each epoch
            std::shuffle(train_examples.begin(), train_examples.end(), rng);

            float epoch_loss = 0.0f;
            size_t epoch_batches = 0;

            spdlog::info("Starting epoch {}/{}", epoch + 1, config.num_epochs);

            // Process batches
            for (size_t batch_start = 0; batch_start < train_examples.size(); batch_start += config.batch_size) {
                if (stop_requested_) {
                    break;
                }

                // Create batch
                size_t batch_end = std::min(batch_start + config.batch_size, train_examples.size());
                std::vector<TrainingExample> batch(train_examples.begin() + batch_start, train_examples.begin() + batch_end);

                // Train on batch
                float batch_loss = trainOnBatch(ctx, batch, config);

                epoch_loss += batch_loss;
                epoch_batches++;

                // Update state
                state_.current_step++;
                state_.current_loss = batch_loss;
                state_.avg_loss = epoch_loss / epoch_batches;

                // Log progress
                if (state_.current_step % config.log_steps == 0) {
                    logTrainingProgress();
                }

                // Save checkpoint if needed
                if (config.save_checkpoints &&
                    config.checkpoint_steps > 0 &&
                    state_.current_step % config.checkpoint_steps == 0) {

                    std::string checkpoint_path = (checkpoint_dir / ("checkpoint_" +
                                                 std::to_string(state_.current_step))).string();

                    std::string description = "Epoch " + std::to_string(epoch + 1) +
                                             "/" + std::to_string(config.num_epochs) +
                                             ", Step " + std::to_string(state_.current_step) +
                                             "/" + std::to_string(state_.total_steps);

                    if (saveCheckpoint(ctx, checkpoint_path, description)) {
                        logCheckpointSaved(checkpoint_path, description);
                    }
                }
            }

            // Log epoch summary
            logEpochSummary();

            // Evaluate on validation set
            if (!val_examples.empty()) {
                float val_loss = evaluateModel(ctx, val_path, config);
                logEvaluationResult(val_loss);

                // Save best model if it has improved
                if (val_loss < state_.best_val_loss) {
                    state_.best_val_loss = val_loss;

                    if (config.save_best_model) {
                        std::string best_model_path = (checkpoint_dir / "best_model").string();
                        std::string description = "Best validation model: " +
                                                 std::to_string(val_loss) + " loss";

                        saveCheckpoint(ctx, best_model_path, description);
                    }
                }
            }
        }

        // Save the final model
        if (!stop_requested_ || config.save_on_interrupt) {
            saveFineTunedModel(ctx, model_info, dataset_info, config);
        }

    } catch (const std::exception& e) {
        spdlog::error("Exception during training: {}", e.what());
        logTrainingError(e.what());
    }

    // Clean up
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_free_model(model);
    }

    spdlog::info("Training completed");
}

float FineTuningEngine::trainOnBatch(
    llama_context* ctx,
    const std::vector<TrainingExample>& batch,
    const FineTuneConfig& config) {

    if (!ctx || batch.empty()) {
        return 0.0f;
    }

    float batch_loss = 0.0f;
    auto batch_start_time = std::chrono::steady_clock::now();

    // Process each example in the batch
    for (const auto& example : batch) {
        const auto& tokens = example.tokens;
        if (tokens.empty()) {
            continue;
        }

        int result = 0;
        if (config.method == FineTuneMethod::FULL) {
            // Full fine-tuning
            result = llama_model_train(
                llama_get_model(ctx),
                ctx,
                reinterpret_cast<const float*>(tokens.data()),
                tokens.size(),
                1,  // batch size of 1 for token-by-token processing
                config.learning_rate,
                false,  // not using LoRA
                0       // LoRA rank not used
            );
        } else {
            // LoRA/QLoRA
            result = llama_model_train(
                llama_get_model(ctx),
                ctx,
                reinterpret_cast<const float*>(tokens.data()),
                tokens.size(),
                1,  // batch size of 1 for token-by-token processing
                config.learning_rate,
                true,  // using LoRA
                config.lora_rank
            );
        }

        if (result != 0) {
            spdlog::warn("Training batch returned error code: {}", result);
        }

        // Accumulate loss
        float example_loss = llama_get_loss(ctx);
        batch_loss += example_loss;

        // Accumulate token count for throughput calculation
        state_.total_tokens_processed += tokens.size();
    }

    // Calculate average loss for the batch
    float avg_batch_loss = batch_loss / batch.size();

    // Calculate throughput
    auto batch_end_time = std::chrono::steady_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        batch_end_time - batch_start_time).count();

    size_t total_tokens = 0;
    for (const auto& example : batch) {
        total_tokens += example.tokens.size();
    }

    float tokens_per_second = (batch_duration > 0)
        ? (1000.0f * total_tokens / batch_duration)
        : 0.0f;

    state_.tokens_per_second = tokens_per_second;

    return avg_batch_loss;
}

float FineTuningEngine::evaluateModel(
    llama_context* ctx,
    const fs::path& val_path,
    const FineTuneConfig& config) {

    if (!ctx || !fs::exists(val_path)) {
        return 0.0f;
    }

    spdlog::info("Evaluating model on validation set");

    std::vector<TrainingExample> val_examples;

    // Load validation examples from file
    std::ifstream val_file(val_path, std::ios::binary);
    if (!val_file.is_open()) {
        spdlog::error("Failed to open validation file: {}", val_path.string());
        return 0.0f;
    }

    size_t num_examples = 0;
    val_file.read(reinterpret_cast<char*>(&num_examples), sizeof(size_t));

    val_examples.reserve(num_examples);

    for (size_t i = 0; i < num_examples; ++i) {
        size_t len = 0;
        val_file.read(reinterpret_cast<char*>(&len), sizeof(size_t));

        TrainingExample example;
        example.tokens.resize(len);
        val_file.read(reinterpret_cast<char*>(example.tokens.data()), len * sizeof(llama_token));

        val_examples.push_back(std::move(example));
    }

    val_file.close();

    // Evaluate model on validation examples
    float total_loss = 0.0f;
    size_t total_tokens = 0;

    for (const auto& example : val_examples) {
        const auto& tokens = example.tokens;
        if (tokens.empty()) {
            continue;
        }

        // Feed tokens to context without training
        if (llama_eval(ctx, tokens.data(), tokens.size(), 0, nullptr) != 0) {
            spdlog::warn("Evaluation returned error code");
            continue;
        }

        // Get perplexity (loss) for the example
        float example_loss = llama_get_loss(ctx);
        total_loss += example_loss * tokens.size();  // Weight by sequence length
        total_tokens += tokens.size();
    }

    // Calculate weighted average loss
    float avg_loss = (total_tokens > 0) ? (total_loss / total_tokens) : 0.0f;

    spdlog::info("Validation loss: {:.4f} over {} tokens", avg_loss, total_tokens);

    return avg_loss;
}

bool FineTuningEngine::saveCheckpoint(
    llama_context* ctx,
    const fs::path& path,
    const std::string& description) {

    if (!ctx) {
        return false;
    }

    spdlog::info("Saving checkpoint to '{}'", path.string());

    try {
        // Create directory if it doesn't exist
        fs::create_directories(path.parent_path());

        // Save adapter weights (for LoRA/QLoRA) or full model
        int result = llama_save_adapter(ctx, path.string().c_str());

        if (result != 0) {
            spdlog::error("Failed to save checkpoint: error code {}", result);
            return false;
        }

        // Save metadata
        std::ofstream meta_file(path.string() + ".meta", std::ios::out);
        if (meta_file.is_open()) {
            meta_file << "timestamp: " << getCurrentTimestamp() << "\n";
            meta_file << "description: " << description << "\n";
            meta_file << "epoch: " << state_.current_epoch << "\n";
            meta_file << "step: " << state_.current_step << "\n";
            meta_file << "loss: " << state_.current_loss << "\n";
            meta_file.close();
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Exception during checkpoint saving: {}", e.what());
        return false;
    }
}

bool FineTuningEngine::saveFineTunedModel(
    llama_context* ctx,
    const ModelInfo& base_model,
    const DatasetInfo& dataset,
    const FineTuneConfig& config) {

    if (!ctx) {
        return false;
    }

    std::string timestamp = getCurrentTimestamp(true);
    std::string method_str;

    switch (config.method) {
        case FineTuneMethod::LORA:
            method_str = "lora";
            break;
        case FineTuneMethod::QLORA:
            method_str = "qlora";
            break;
        case FineTuneMethod::FULL:
            method_str = "full";
            break;
        default:
            method_str = "unknown";
            break;
    }

    std::string new_model_id = base_model.id + "-ft-" +
                              method_str + "-" +
                              dataset.id + "-" +
                              timestamp;

    // Create model directory
    fs::path model_dir = model_manager_.getBaseDir() / "models" / new_model_id;
    fs::create_directories(model_dir);

    spdlog::info("Saving fine-tuned model to '{}'", model_dir.string());

    try {
        // Save the adapter or full model
        fs::path model_path = model_dir / "model.bin";
        int result = llama_save_adapter(ctx, model_path.string().c_str());

        if (result != 0) {
            spdlog::error("Failed to save fine-tuned model: error code {}", result);
            return false;
        }

        // Create model info
        ModelInfo new_model;
        new_model.id = new_model_id;
        new_model.base_model = base_model.id;
        new_model.description = "Fine-tuned from " + base_model.id + " using " +
                              dataset.id + " dataset with " + method_str;
        new_model.fine_tuned = true;
        new_model.fine_tune_method = config.method;
        new_model.parameters = base_model.parameters;  // Same parameter count as base model
        new_model.quantization = config.method == FineTuneMethod::QLORA ?
                               config.quantization_type : QuantizationType::NONE;
        new_model.context_length = config.context_length;
        new_model.created_at = getCurrentTimestamp();

        // Create config file
        fs::path config_path = model_dir / "config.json";
        std::ofstream config_file(config_path, std::ios::out);
        if (config_file.is_open()) {
            // Use nlohmann::json to create JSON file
            nlohmann::json j = {
                {"id", new_model.id},
                {"base_model", new_model.base_model},
                {"description", new_model.description},
                {"fine_tuned", new_model.fine_tuned},
                {"fine_tune_method", static_cast<int>(new_model.fine_tune_method)},
                {"parameters", new_model.parameters},
                {"quantization", static_cast<int>(new_model.quantization)},
                {"context_length", new_model.context_length},
                {"created_at", new_model.created_at},
                {"dataset_id", dataset.id},
                {"training_config", {
                    {"method", static_cast<int>(config.method)},
                    {"learning_rate", config.learning_rate},
                    {"num_epochs", config.num_epochs},
                    {"batch_size", config.batch_size},
                    {"lora_rank", config.lora_rank},
                    {"lora_alpha", config.lora_alpha},
                    {"lora_dropout", config.lora_dropout},
                    {"validation_split", config.validation_split},
                    {"seed", config.seed}
                }}
            };

            config_file << j.dump(4);
            config_file.close();
        }

        // Save training stats
        fs::path stats_path = model_dir / "training_stats.json";
        std::ofstream stats_file(stats_path, std::ios::out);
        if (stats_file.is_open()) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                state_.end_time - state_.start_time).count();

            nlohmann::json j = {
                {"total_epochs", state_.current_epoch + 1},
                {"total_steps", state_.current_step},
                {"final_loss", state_.current_loss},
                {"best_val_loss", state_.best_val_loss},
                {"total_tokens_processed", state_.total_tokens_processed},
                {"tokens_per_second", state_.tokens_per_second},
                {"training_duration_seconds", duration},
                {"log_file", log_file_path_}
            };

            stats_file << j.dump(4);
            stats_file.close();
        }

        // Register the model in the model manager
        if (!model_manager_.addModel(new_model)) {
            spdlog::error("Failed to register fine-tuned model in model manager");
            return false;
        }

        spdlog::info("Successfully saved fine-tuned model '{}'", new_model_id);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception during fine-tuned model saving: {}", e.what());
        return false;
    }
}

std::string FineTuningEngine::getCurrentTimestamp(bool filename_safe) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    if (filename_safe) {
        std::tm tm_now;
#ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_now);
#endif
        ss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    } else {
        std::tm tm_now;
#ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_now);
#endif
        ss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
    }

    return ss.str();
}

void FineTuningEngine::logTrainingConfig(
    const ModelInfo& model_info,
    const DatasetInfo& dataset_info,
    const FineTuneConfig& config) {

    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "=== Training Configuration ===" << std::endl;
    log_file_ << "Timestamp: " << getCurrentTimestamp() << std::endl;
    log_file_ << "Model: " << model_info.id << std::endl;
    log_file_ << "Dataset: " << dataset_info.id << std::endl;

    log_file_ << "Method: ";
    switch (config.method) {
        case FineTuneMethod::LORA:
            log_file_ << "LoRA" << std::endl;
            break;
        case FineTuneMethod::QLORA:
            log_file_ << "QLoRA" << std::endl;
            break;
        case FineTuneMethod::FULL:
            log_file_ << "Full Fine-tuning" << std::endl;
            break;
        default:
            log_file_ << "Unknown" << std::endl;
            break;
    }

    if (config.method == FineTuneMethod::LORA || config.method == FineTuneMethod::QLORA) {
        log_file_ << "LoRA Rank: " << config.lora_rank << std::endl;
        log_file_ << "LoRA Alpha: " << config.lora_alpha << std::endl;
        log_file_ << "LoRA Dropout: " << config.lora_dropout << std::endl;
    }

    if (config.method == FineTuneMethod::QLORA) {
        log_file_ << "Quantization: ";
        switch (config.quantization_type) {
            case QuantizationType::Q4_0:
                log_file_ << "Q4_0" << std::endl;
                break;
            case QuantizationType::Q4_1:
                log_file_ << "Q4_1" << std::endl;
                break;
            case QuantizationType::Q5_0:
                log_file_ << "Q5_0" << std::endl;
                break;
            case QuantizationType::Q5_1:
                log_file_ << "Q5_1" << std::endl;
                break;
            case QuantizationType::Q8_0:
                log_file_ << "Q8_0" << std::endl;
                break;
            default:
                log_file_ << "Unknown" << std::endl;
                break;
        }
    }

    log_file_ << "Learning Rate: " << config.learning_rate << std::endl;
    log_file_ << "Epochs: " << config.num_epochs << std::endl;
    log_file_ << "Batch Size: " << config.batch_size << std::endl;
    log_file_ << "Context Length: " << config.context_length << std::endl;
    log_file_ << "Validation Split: " << config.validation_split << std::endl;
    log_file_ << "Seed: " << config.seed << std::endl;
    log_file_ << "Save Checkpoints: " << (config.save_checkpoints ? "Yes" : "No") << std::endl;

    if (config.save_checkpoints) {
        log_file_ << "Checkpoint Steps: " << config.checkpoint_steps << std::endl;
    }

    log_file_ << "Save Best Model: " << (config.save_best_model ? "Yes" : "No") << std::endl;
    log_file_ << "==========================" << std::endl << std::endl;

    log_file_.flush();
}

void FineTuningEngine::logTrainingProgress() {
    if (!log_file_.is_open()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - state_.start_time).count();

    // Calculate ETA
    float progress = static_cast<float>(state_.current_step) / state_.total_steps;
    int64_t eta_seconds = (progress > 0.01f) ?
                        static_cast<int64_t>(duration / progress - duration) : 0;

    log_file_ << "Step " << state_.current_step << "/" << state_.total_steps
              << " (Epoch " << state_.current_epoch + 1 << "/" << state_.config.num_epochs << ")"
              << " - Loss: " << std::fixed << std::setprecision(6) << state_.current_loss
              << " - Avg Loss: " << std::fixed << std::setprecision(6) << state_.avg_loss
              << " - Tokens/sec: " << std::fixed << std::setprecision(2) << state_.tokens_per_second
              << " - Duration: " << formatDuration(duration)
              << " - ETA: " << formatDuration(eta_seconds)
              << std::endl;

    log_file_.flush();

    // Also log to console
    spdlog::info("Training progress - Step: {}/{}, Epoch: {}/{}, Loss: {:.6f}, Tokens/sec: {:.2f}, ETA: {}",
                state_.current_step, state_.total_steps,
                state_.current_epoch + 1, state_.config.num_epochs,
                state_.current_loss, state_.tokens_per_second,
                formatDuration(eta_seconds));
}

void FineTuningEngine::logEvaluationResult(float val_loss) {
    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "Validation - "
              << "Epoch " << state_.current_epoch + 1 << "/" << state_.config.num_epochs
              << " - Val Loss: " << std::fixed << std::setprecision(6) << val_loss;

    if (val_loss < state_.best_val_loss) {
        log_file_ << " (New best)";
        state_.best_val_loss = val_loss;
    }

    log_file_ << std::endl;
    log_file_.flush();

    // Also log to console
    spdlog::info("Validation - Epoch: {}/{}, Val Loss: {:.6f}{}",
                state_.current_epoch + 1, state_.config.num_epochs,
                val_loss, (val_loss < state_.best_val_loss ? " (New best)" : ""));
}

void FineTuningEngine::logEpochSummary() {
    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "Epoch " << state_.current_epoch + 1 << "/" << state_.config.num_epochs
              << " completed - Avg Loss: " << std::fixed << std::setprecision(6) << state_.avg_loss
              << std::endl;

    log_file_.flush();

    // Also log to console
    spdlog::info("Epoch {}/{} completed - Avg Loss: {:.6f}",
                state_.current_epoch + 1, state_.config.num_epochs,
                state_.avg_loss);
}

void FineTuningEngine::logCheckpointSaved(const std::string& path, const std::string& description) {
    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "Checkpoint saved to '" << path << "' - " << description << std::endl;
    log_file_.flush();

    // Also log to console
    spdlog::info("Checkpoint saved to '{}' - {}", path, description);
}

void FineTuningEngine::logTrainingError(const std::string& error_message) {
    if (!log_file_.is_open()) {
        return;
    }

    log_file_ << "ERROR: " << error_message << std::endl;
    log_file_.flush();

    // Also log to console
    spdlog::error("Training error: {}", error_message);
}

// Utility function to format duration in hours:minutes:seconds
std::string formatDuration(int64_t seconds) {
    if (seconds < 0) {
        return "unknown";
    }

    int64_t hours = seconds / 3600;
    int64_t minutes = (seconds % 3600) / 60;
    int64_t secs = seconds % 60;

    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (hours > 0 || minutes > 0) {
        ss << minutes << "m ";
    }
    ss << secs << "s";

    return ss.str();
}

} // namespace localllm
