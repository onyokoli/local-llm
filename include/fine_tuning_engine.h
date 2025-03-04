#ifndef LOCALLLM_FINE_TUNING_ENGINE_H
#define LOCALLLM_FINE_TUNING_ENGINE_H

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <chrono>

#include "types.h"

// Forward declarations
struct llama_context;

namespace localllm {

class ModelManager;
class DatasetManager;

class FineTuningEngine {
public:
    FineTuningEngine(ModelManager& model_manager, DatasetManager& dataset_manager);
    ~FineTuningEngine();

    // Training state tracking
    struct TrainingState {
        std::string model_id;
        std::string dataset_id;
        FineTuneConfig config;

        int current_epoch = 0;
        int current_step = 0;
        int total_steps = 0;
        float current_loss = 0.0f;
        float avg_loss = 0.0f;
        float best_val_loss = std::numeric_limits<float>::max();

        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        // Training throughput metrics
        float tokens_per_second = 0.0f;
        uint64_t total_tokens_processed = 0;
    };

    // Start fine-tuning a model with a dataset
    bool startFineTuning(const std::string& model_id,
                        const std::string& dataset_id,
                        const FineTuneConfig& config = FineTuneConfig());

    // Stop current training process
    void stopTraining(bool wait_for_completion = false);

    // Check if training is in progress
    bool isTraining() const;

    // Get current training state
    TrainingState getTrainingState() const;

    // Get path to current log file
    std::string getLogFilePath() const;

private:
    ModelManager& model_manager_;
    DatasetManager& dataset_manager_;
    std::atomic<bool> is_training_{false};
    std::atomic<bool> stop_requested_{false};
    std::mutex training_mutex_;
    std::condition_variable training_cv_;

    // Training state
    TrainingState state_;

    // For progress tracking
    std::string log_file_path_;
    std::ofstream log_file_;
    std::thread training_thread_;

    // Main training loop
    void runTrainingLoop(const ModelInfo& model_info,
                        const DatasetInfo& dataset_info,
                        const FineTuneConfig& config);

    // Training and evaluation methods
    float trainOnBatch(llama_context* ctx, const std::vector<TrainingExample>& batch, const FineTuneConfig& config);
    float evaluateModel(llama_context* ctx, const fs::path& val_path, const FineTuneConfig& config);

    // Checkpoint and model saving
    bool saveCheckpoint(llama_context* ctx, const fs::path& path, const std::string& description);
    bool saveFineTunedModel(llama_context* ctx,
                          const ModelInfo& base_model,
                          const DatasetInfo& dataset,
                          const FineTuneConfig& config);

    // Utility methods
    std::string getCurrentTimestamp(bool filename_safe = false);

    // Logging methods
    void logTrainingConfig(const ModelInfo& model_info,
                          const DatasetInfo& dataset_info,
                          const FineTuneConfig& config);
    void logTrainingProgress();
    void logEvaluationResult(float val_loss);
    void logEpochSummary();
    void logCheckpointSaved(const std::string& path, const std::string& description);
    void logTrainingError(const std::string& error_message);
};

} // namespace localllm

#endif // LOCALLLM_FINE_TUNING_ENGINE_H
