#ifndef LOCALLLM_MODEL_MANAGER_H
#define LOCALLLM_MODEL_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <mutex>

#include "types.h"

// Forward declaration for sqlite3
struct sqlite3;

namespace localllm {

class ModelManager {
public:
    ModelManager(const fs::path& models_dir);
    ~ModelManager();

    // Scan filesystem for models and update the internal database
    void scanAvailableModels();

    // Add a model to the database
    bool addModel(const ModelInfo& info);

    // Update model metadata
    bool updateModel(const ModelInfo& info);

    // Remove a model (optionally delete files)
    bool removeModel(const std::string& id, bool delete_files = false);

    // Get model information
    std::optional<ModelInfo> getModelInfo(const std::string& id) const;

    // List all available models
    std::vector<ModelInfo> listAvailableModels() const;

    // Check if a model exists
    bool isModelAvailable(const std::string& id) const;

    // Download a model from a source (e.g., HuggingFace)
    bool downloadModel(const std::string& model_id, const std::string& source = "huggingface",
                      bool quantize = false, const std::string& quantization_type = "q4_0");

    // Get the filesystem path for a model
    fs::path getModelPath(const std::string& id) const;

private:
    fs::path models_dir_;                    // Base directory for storing models
    fs::path metadata_db_path_;              // Path to SQLite database for metadata
    sqlite3* metadata_db_ = nullptr;         // SQLite database handle
    std::unordered_map<std::string, ModelInfo> available_models_;  // Cache of available models
    mutable std::mutex mutex_;               // Mutex for thread safety

    // Initialize SQLite database for model metadata
    void initMetadataDB();
};

} // namespace localllm

#endif // LOCALLLM_MODEL_MANAGER_H
