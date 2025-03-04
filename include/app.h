#ifndef LOCALLLM_APP_H
#define LOCALLLM_APP_H

#include <string>
#include <vector>
#include <optional>
#include <filesystem>

#include "types.h"

namespace localllm {

// Forward declarations for classes used with unique_ptr
class ModelManager;
class DatasetManager;
class FineTuningEngine;
class InferenceServer;

class LocalLLMApp {
public:
    LocalLLMApp(const fs::path& base_dir, uint16_t server_port = 8080);
    ~LocalLLMApp();  // Must declare the destructor for unique_ptr with incomplete types

    // Initialize the application
    void initialize();

    // Model Management
    bool downloadModel(const std::string& model_id, const std::string& source = "huggingface",
                      bool quantize = false, const std::string& quantization_type = "q4_0");
    std::vector<ModelInfo> listModels();
    std::optional<ModelInfo> getModelInfo(const std::string& id);
    bool removeModel(const std::string& id, bool delete_files = false);

    // Dataset Management
    std::string importDataset(const fs::path& path, const std::string& name = "",
                            const std::string& format = "jsonl");
    std::vector<DatasetInfo> listDatasets();
    std::optional<DatasetInfo> getDatasetInfo(const std::string& id);

    // Fine-tuning
    bool startFineTuning(const std::string& model_id, const std::string& dataset_id,
                        const FineTuneConfig& config = FineTuneConfig());
    void stopFineTuning(bool wait_for_completion = false);
    bool isFineTuningActive();

    // Inference
    bool startInferenceServer();
    void stopInferenceServer();

    // Utility methods
    std::string getVersionInfo();
    void shutdown();

private:
    fs::path base_dir_;
    std::unique_ptr<ModelManager> model_manager_;
    std::unique_ptr<DatasetManager> dataset_manager_;
    std::unique_ptr<FineTuningEngine> fine_tuning_engine_;
    std::unique_ptr<InferenceServer> inference_server_;
};

} // namespace localllm

#endif // LOCALLLM_APP_H
