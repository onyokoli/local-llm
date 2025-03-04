#ifndef LOCALLLM_DATASET_MANAGER_H
#define LOCALLLM_DATASET_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <fstream>
#include <mutex>

#include "types.h"

// Forward declarations
struct llama_context;
struct sqlite3;

namespace localllm {

class DatasetManager {
public:
    DatasetManager(const fs::path& datasets_dir);
    ~DatasetManager();

    // Scan filesystem for datasets and update the internal database
    void scanAvailableDatasets();

    // Import a dataset from a file
    std::string importDataset(const fs::path& source_path, const std::string& name, const std::string& format_type = "jsonl");

    // Analyze a dataset to extract metadata
    bool analyzeDataset(DatasetInfo& info);

    // Preprocess a dataset for training
    bool preprocessDataset(const std::string& id, llama_context* ctx);

    // Get dataset information
    std::optional<DatasetInfo> getDatasetInfo(const std::string& id) const;

    // List all available datasets
    std::vector<DatasetInfo> listAvailableDatasets() const;

    // Check if a dataset exists
    bool isDatasetAvailable(const std::string& id) const;

private:
    fs::path datasets_dir_;                    // Base directory for storing datasets
    fs::path metadata_db_path_;                // Path to SQLite database for metadata
    sqlite3* metadata_db_ = nullptr;           // SQLite database handle
    std::unordered_map<std::string, DatasetInfo> available_datasets_;  // Cache of available datasets
    mutable std::mutex mutex_;                 // Mutex for thread safety

    // Initialize SQLite database for dataset metadata
    void initMetadataDB();

    // Analyze different dataset formats
    bool analyzeJsonlDataset(DatasetInfo& info);
    bool analyzeTextDataset(DatasetInfo& info);
    bool analyzeCsvDataset(DatasetInfo& info);

    // Preprocess different dataset formats
    bool preprocessJsonlDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx);
    bool preprocessTextDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx);
    bool preprocessCsvDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx);

    // Helper methods
    std::vector<llama_token> tokenize(llama_context* ctx, const std::string& text, bool add_bos);
    void writeTokenizedExample(std::ofstream& file, const std::vector<llama_token>& input_tokens,
                             const std::vector<llama_token>& output_tokens);
    uint64_t estimateTokenCount(const std::string& text);
    std::string generateUniqueId();
    std::string replaceAll(std::string str, const std::string& from, const std::string& to);

    // Database operations
    bool addDatasetToDb(const DatasetInfo& info);
    bool updateDatasetInDb(const DatasetInfo& info);
};

} // namespace localllm

#endif // LOCALLLM_DATASET_MANAGER_H
