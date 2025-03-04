#include "model_manager.h"
#include <sqlite3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <regex>
#include <thread>
#include <future>
#include <atomic>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Replacement for C++20's ends_with
static bool endsWith(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

namespace localllm {

// Helper function for CURL write callbacks
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    auto* mem = static_cast<std::string*>(userp);
    mem->append(static_cast<char*>(contents), realsize);
    return realsize;
}

// Helper function for downloading files with progress reporting
static size_t DownloadCallback(void* ptr, size_t size, size_t nmemb, void* data) {
    FILE* file = static_cast<FILE*>(data);
    return fwrite(ptr, size, nmemb, file);
}

// Helper function for progress updates
static int ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                           [[maybe_unused]] curl_off_t ultotal,
                           [[maybe_unused]] curl_off_t ulnow) {
    if (dltotal == 0) return 0;

    int percentage = static_cast<int>((dlnow * 100) / dltotal);
    auto* lastPercentage = static_cast<int*>(clientp);

    // Only update when percentage changes significantly to avoid console spam
    if (percentage >= *lastPercentage + 5 || percentage == 100) {
        *lastPercentage = percentage;
        spdlog::info("Download progress: {}%", percentage);
    }

    return 0;
}

ModelManager::ModelManager(const fs::path& models_dir)
    : models_dir_(models_dir), metadata_db_path_(models_dir / "models_metadata.db") {

    try {
        // Create models directory if it doesn't exist
        if (!fs::exists(models_dir_)) {
            spdlog::info("Creating models directory at {}", models_dir_.string());
            fs::create_directories(models_dir_);
        }

        // Initialize SQLite database
        initMetadataDB();

        // Initialize CURL globally (will be cleaned up in destructor)
        curl_global_init(CURL_GLOBAL_DEFAULT);

        // Scan for existing models
        scanAvailableModels();

    } catch (const std::exception& e) {
        spdlog::error("Error initializing ModelManager: {}", e.what());
        throw;
    }
}

ModelManager::~ModelManager() {
    // Close SQLite database connection
    if (metadata_db_) {
        sqlite3_close(metadata_db_);
        metadata_db_ = nullptr;
    }

    // Cleanup CURL
    curl_global_cleanup();
}

void ModelManager::initMetadataDB() {
    int rc = sqlite3_open(metadata_db_path_.c_str(), &metadata_db_);

    if (rc) {
        std::string error_msg = sqlite3_errmsg(metadata_db_);
        sqlite3_close(metadata_db_);
        metadata_db_ = nullptr;
        throw std::runtime_error("Cannot open database: " + error_msg);
    }

    // Create models table if it doesn't exist
    const char* create_table_sql =
        "CREATE TABLE IF NOT EXISTS models ("
        "id TEXT PRIMARY KEY,"
        "name TEXT NOT NULL,"
        "source TEXT NOT NULL,"
        "architecture TEXT NOT NULL,"
        "parameter_count INTEGER NOT NULL,"
        "quantization_type TEXT,"
        "date_added TEXT NOT NULL,"
        "last_modified TEXT NOT NULL,"
        "size_bytes INTEGER NOT NULL,"
        "is_fine_tuned INTEGER NOT NULL,"
        "parent_model TEXT,"
        "description TEXT,"
        "metadata TEXT"
        ");";

    char* err_msg = nullptr;
    rc = sqlite3_exec(metadata_db_, create_table_sql, nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        std::string error = err_msg;
        sqlite3_free(err_msg);
        throw std::runtime_error("SQL error: " + error);
    }

    spdlog::debug("Model metadata database initialized");
}

void ModelManager::scanAvailableModels() {
    std::lock_guard<std::mutex> lock(mutex_);

    available_models_.clear();

    // First, get all models from database
    const char* query = "SELECT * FROM models;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(metadata_db_, query, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
        return;
    }

    // Read all models from database and add to cache
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ModelInfo info;

        info.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        info.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        // Map 'source' field (not in ModelInfo)
        std::string source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        // Map 'architecture' to 'architecture_type'
        info.architecture_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        info.parameter_count = sqlite3_column_int64(stmt, 4);

        if (sqlite3_column_type(stmt, 5) != SQLITE_NULL) {
            // Map 'quantization_type' to 'quantization_method'
            info.quantization_method = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        }

        // Map 'date_added' field (not in ModelInfo)
        std::string date_added = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        // Map 'last_modified' field (not in ModelInfo)
        std::string last_modified = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        // Map 'size_bytes' field (not in ModelInfo) - not used in this function
        sqlite3_column_int64(stmt, 8);
        // Map 'is_fine_tuned' to 'is_finetuned'
        info.is_finetuned = sqlite3_column_int(stmt, 9) != 0;

        if (sqlite3_column_type(stmt, 10) != SQLITE_NULL) {
            // Map 'parent_model' to 'parent_model_id'
            info.parent_model_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        }

        if (sqlite3_column_type(stmt, 11) != SQLITE_NULL) {
            // Map 'description' field (not in ModelInfo)
            // Not stored in ModelInfo, just reading from DB
            std::string description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        }

        if (sqlite3_column_type(stmt, 12) != SQLITE_NULL) {
            try {
                const char* json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12));
                // Map 'metadata' field (not in ModelInfo)
                // Not stored in ModelInfo, just reading from DB
                json metadata = json::parse(json_str);
            } catch (const json::exception& e) {
                spdlog::warn("Error parsing JSON metadata for model {}: {}", info.id, e.what());
            }
        }

        // Check if model files actually exist
        fs::path model_path = getModelPath(info.id);
        if (fs::exists(model_path)) {
            available_models_[info.id] = info;
        } else {
            spdlog::warn("Model files for {} not found at {}. Removing from database.",
                         info.id, model_path.string());
            removeModel(info.id, false);
        }
    }

    sqlite3_finalize(stmt);

    // Now scan filesystem for models that might not be in the database
    try {
        for (const auto& entry : fs::directory_iterator(models_dir_)) {
            if (entry.is_directory()) {
                std::string dir_name = entry.path().filename().string();

                // Skip special directories
                if (dir_name == "." || dir_name == "..") {
                    continue;
                }

                // Check if this directory contains model files
                bool has_model_files = false;
                for (const auto& file : fs::directory_iterator(entry.path())) {
                    if (file.path().extension() == ".bin" ||
                        file.path().extension() == ".gguf" ||
                        file.path().extension() == ".ggml") {
                        has_model_files = true;
                        break;
                    }
                }

                if (has_model_files && !available_models_.count(dir_name)) {
                    spdlog::info("Found model directory not in database: {}", dir_name);

                    // Try to infer model info from directory structure and filenames
                    ModelInfo info;
                    info.id = dir_name;

                    // Parse the model ID to extract information
                    std::regex pattern("([^/]+?)(?:-([^-]+))?(?:-ft-([0-9_]+))?$");
                    std::smatch match;
                    if (std::regex_match(dir_name, match, pattern)) {
                        info.name = match[1].str();

                        if (match[2].matched) {
                            // Map 'quantization_type' to 'quantization_method'
                            info.quantization_method = match[2].str();
                        }

                        if (match[3].matched) {
                            // Map 'is_fine_tuned' to 'is_finetuned'
                            info.is_finetuned = true;
                            if (!match[3].str().empty()) {
                                // Extract parent model if this is a fine-tuned model
                                // Map 'parent_model' to 'parent_model_id'
                                info.parent_model_id = match[1].str();
                                if (match[2].matched) {
                                    info.parent_model_id = info.parent_model_id.value() + "-" + match[2].str();
                                }
                            }
                        } else {
                            // Map 'is_fine_tuned' to 'is_finetuned'
                            info.is_finetuned = false;
                        }
                    } else {
                        info.name = dir_name;
                        // Map 'is_fine_tuned' to 'is_finetuned'
                        info.is_finetuned = false;
                    }

                    // Set default values
                    info.architecture_type = "unknown";
                    info.parameter_count = 0;

                    // Get file size
                    size_t total_size = 0;
                    for (const auto& file : fs::recursive_directory_iterator(entry.path())) {
                        if (file.is_regular_file()) {
                            total_size += file.file_size();
                        }
                    }

                    // Try to find a config.json file to extract more information
                    fs::path config_path = entry.path() / "config.json";
                    if (fs::exists(config_path)) {
                        try {
                            std::ifstream config_file(config_path);
                            json config = json::parse(config_file);

                            if (config.contains("architecture")) {
                                info.architecture_type = config["architecture"].get<std::string>();
                            }

                            if (config.contains("model_type")) {
                                info.architecture_type = config["model_type"].get<std::string>();
                            }

                            // Try to determine parameter count
                            if (config.contains("n_params")) {
                                info.parameter_count = config["n_params"].get<int64_t>();
                            } else if (config.contains("num_parameters")) {
                                info.parameter_count = config["num_parameters"].get<int64_t>();
                            } else {
                                // Try to estimate from model size (rough approximation)
                                // Assuming FP16 precision (2 bytes per parameter)
                                if (info.quantization_method.empty()) {
                                    info.parameter_count = total_size / 2;
                                } else if (info.quantization_method == "q4_0" ||
                                           info.quantization_method == "q4_1") {
                                    // 4-bit quantization (0.5 bytes per parameter)
                                    info.parameter_count = total_size * 2;
                                } else if (info.quantization_method == "q5_0" ||
                                           info.quantization_method == "q5_1") {
                                    // 5-bit quantization (0.625 bytes per parameter)
                                    info.parameter_count = total_size * 1.6;
                                } else if (info.quantization_method == "q8_0") {
                                    // 8-bit quantization (1 byte per parameter)
                                    info.parameter_count = total_size;
                                }
                            }
                        } catch (const std::exception& e) {
                            spdlog::warn("Failed to parse config.json for model {}: {}",
                                         dir_name, e.what());
                        }
                    }

                    // Add to database
                    addModel(info);
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        spdlog::error("Filesystem error during model scanning: {}", e.what());
    }

    spdlog::info("Scan complete. Found {} models", available_models_.size());
}

bool ModelManager::addModel(const ModelInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);

    const char* insert_sql =
        "INSERT INTO models (id, name, source, architecture, parameter_count, "
        "quantization_type, date_added, last_modified, size_bytes, is_fine_tuned, "
        "parent_model, description, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(metadata_db_, insert_sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
        return false;
    }

    sqlite3_bind_text(stmt, 1, info.id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, info.name.c_str(), -1, SQLITE_STATIC);

    // Default value for 'source' since it's not in ModelInfo
    std::string source = "unknown";
    sqlite3_bind_text(stmt, 3, source.c_str(), -1, SQLITE_STATIC);

    // Map 'architecture_type' to 'architecture'
    sqlite3_bind_text(stmt, 4, info.architecture_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 5, info.parameter_count);

    // Map 'quantization_method' to 'quantization_type'
    if (info.quantization_method.empty()) {
        sqlite3_bind_null(stmt, 6);
    } else {
        sqlite3_bind_text(stmt, 6, info.quantization_method.c_str(), -1, SQLITE_STATIC);
    }

    // Default values for timestamp fields not in ModelInfo
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    std::string current_time = ss.str();

    sqlite3_bind_text(stmt, 7, current_time.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 8, current_time.c_str(), -1, SQLITE_STATIC);

    // Default value for 'size_bytes'
    size_t size_bytes = 0;
    fs::path model_dir = getModelPath(info.id);
    if (fs::exists(model_dir)) {
        for (const auto& entry : fs::recursive_directory_iterator(model_dir)) {
            if (entry.is_regular_file()) {
                size_bytes += entry.file_size();
            }
        }
    }
    sqlite3_bind_int64(stmt, 9, size_bytes);

    // Map 'is_finetuned' to 'is_fine_tuned'
    sqlite3_bind_int(stmt, 10, info.is_finetuned ? 1 : 0);

    // Map 'parent_model_id' to 'parent_model'
    if (!info.parent_model_id.has_value() || info.parent_model_id->empty()) {
        sqlite3_bind_null(stmt, 11);
    } else {
        sqlite3_bind_text(stmt, 11, info.parent_model_id->c_str(), -1, SQLITE_STATIC);
    }

    // Default value for 'description'
    sqlite3_bind_null(stmt, 12);

    // Empty metadata as JSON string
    json empty_metadata = json::object();
    std::string metadata_json = empty_metadata.dump();
    sqlite3_bind_text(stmt, 13, metadata_json.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        spdlog::error("Failed to insert model: {}", sqlite3_errmsg(metadata_db_));
        return false;
    }

    // Update in-memory cache
    available_models_[info.id] = info;

    spdlog::info("Added model {} to database", info.id);
    return true;
}

// Function to extract model information from a HuggingFace model ID
static bool parseHuggingFaceModelId(const std::string& model_id, ModelInfo& info) {
    // Parse format like "organization/model-name"
    size_t slash_pos = model_id.find('/');

    if (slash_pos == std::string::npos) {
        // No organization specified, use as-is
        info.id = model_id;
        info.name = model_id;
    } else {
        std::string org = model_id.substr(0, slash_pos);
        std::string name = model_id.substr(slash_pos + 1);

        info.id = model_id;
        info.name = name;
    }

    return true;
}

// Function to download model files from HuggingFace
static bool downloadFromHuggingFace(const std::string& model_id, const fs::path& output_dir,
                                   std::string& error_message) {
    std::string api_url = "https://huggingface.co/api/models/" + model_id;
    CURL* curl = curl_easy_init();

    if (!curl) {
        error_message = "Failed to initialize CURL";
        return false;
    }

    std::string response_data;

    curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "LocalLLM/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        error_message = "CURL error: " + std::string(curl_easy_strerror(res));
        return false;
    }

    // Parse JSON response
    json model_info;
    try {
        model_info = json::parse(response_data);
    } catch (const json::exception& e) {
        error_message = "Failed to parse API response: " + std::string(e.what());
        return false;
    }

    // Check if model exists
    if (model_info.contains("error")) {
        error_message = "API error: " + model_info["error"].get<std::string>();
        return false;
    }

    // Get files list
    std::string files_url = "https://huggingface.co/api/models/" + model_id + "/tree";
    curl = curl_easy_init();

    if (!curl) {
        error_message = "Failed to initialize CURL";
        return false;
    }

    response_data.clear();

    curl_easy_setopt(curl, CURLOPT_URL, files_url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "LocalLLM/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        error_message = "CURL error: " + std::string(curl_easy_strerror(res));
        return false;
    }

    // Parse files list
    json files_list;
    try {
        files_list = json::parse(response_data);
    } catch (const json::exception& e) {
        error_message = "Failed to parse files list: " + std::string(e.what());
        return false;
    }

    if (!files_list.is_array()) {
        error_message = "Invalid files list format";
        return false;
    }

    // Filter to get only model files and config
    std::vector<std::string> model_files;
    bool has_config = false;

    for (const auto& file : files_list) {
        if (!file.contains("path") || !file["path"].is_string()) {
            continue;
        }

        std::string path = file["path"];

        // Look for model files with ggml, gguf, or bin extensions
        if (endsWith(path, ".bin") || endsWith(path, ".gguf") || endsWith(path, ".ggml") ||
            path == "config.json" || path == "tokenizer.json" || path == "tokenizer_config.json") {

            model_files.push_back(path);

            if (path == "config.json") {
                has_config = true;
            }
        }
    }

    if (model_files.empty()) {
        error_message = "No compatible model files found";
        return false;
    }

    // Create output directory
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Download each file
    spdlog::info("Found {} files to download", model_files.size());

    // Track overall progress
    std::atomic<int> downloaded_files{0};
    int total_files = model_files.size();

    // Set up thread pool for parallel downloads
    const size_t max_threads = std::min<size_t>(4, std::thread::hardware_concurrency());
    std::vector<std::future<bool>> download_futures;

    // Mutex for thread-safe progress reporting
    std::mutex progress_mutex;

    // Download function for each file
    auto download_file = [&](const std::string& file_path) -> bool {
        std::string file_url = "https://huggingface.co/"+model_id+"/resolve/main/"+file_path;
        fs::path output_path = output_dir / file_path;

        // Create directories if needed
        if (fs::path parent_path = output_path.parent_path(); !fs::exists(parent_path)) {
            fs::create_directories(parent_path);
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            return false;
        }

        FILE* fp = fopen(output_path.c_str(), "wb");
        if (!fp) {
            curl_easy_cleanup(curl);
            return false;
        }

        int progress_percentage = 0;

        curl_easy_setopt(curl, CURLOPT_URL, file_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DownloadCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "LocalLLM/1.0");
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L); // 10 minute timeout for large files
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_percentage);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

        CURLcode res = curl_easy_perform(curl);

        fclose(fp);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            spdlog::error("Failed to download {}: {}", file_path, curl_easy_strerror(res));
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            downloaded_files++;
            spdlog::info("Downloaded file {}/{}: {}", downloaded_files.load(), total_files, file_path);
        }

        return true;
    };

    // Start downloads
    for (const auto& file : model_files) {
        if (download_futures.size() >= max_threads) {
            // Wait for one thread to finish before starting a new one
            for (auto it = download_futures.begin(); it != download_futures.end(); ++it) {
                if (it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    if (!it->get()) {
                        // Failed to download a file
                        error_message = "Failed to download one or more files";
                        return false;
                    }
                    download_futures.erase(it);
                    break;
                }
            }

            // If we couldn't find a finished thread, wait for the first one
                        if (download_futures.size() >= max_threads) {
                            if (!download_futures[0].get()) {
                                error_message = "Failed to download one or more files";
                                return false;
                            }
                            download_futures.erase(download_futures.begin());
                        }
                    }

                    download_futures.push_back(std::async(std::launch::async, download_file, file));
                }

                // Wait for remaining downloads to finish
                for (auto& future : download_futures) {
                    if (!future.get()) {
                        error_message = "Failed to download one or more files";
                        return false;
                    }
                }

                spdlog::info("All files downloaded successfully");
                return true;
            }

            // Function to quantize a model
            static bool quantizeModel(const fs::path& model_path, const std::string& quantization_type,
                                     [[maybe_unused]] std::string& error_message) {
                // For now, we'll just simulate quantization with a log message
                // In a real implementation, this would call into llama.cpp or similar quantization tools
                spdlog::info("Quantizing model at {} to {} format", model_path.string(), quantization_type);

                // TODO: Implement actual quantization using llama.cpp or similar tools
                // This is just a placeholder

                return true;
            }

            bool ModelManager::downloadModel(const std::string& model_id, const std::string& source,
                                           bool quantize, const std::string& quantization_type) {
                spdlog::info("Downloading model {} from {} (quantize: {})",
                             model_id, source, quantize ? "true" : "false");

                // Initialize model info
                ModelInfo info;

                // Check if model already exists
                if (isModelAvailable(model_id)) {
                    spdlog::warn("Model {} already exists", model_id);
                    return false;
                }

                // Parse model ID based on source
                if (source == "huggingface") {
                    if (!parseHuggingFaceModelId(model_id, info)) {
                        spdlog::error("Failed to parse HuggingFace model ID: {}", model_id);
                        return false;
                    }
                } else {
                    spdlog::error("Unsupported model source: {}", source);
                    return false;
                }

                // Set up output directory
                fs::path output_dir = getModelPath(model_id);

                // Download model based on source
                std::string error_message;
                bool download_success = false;

                if (source == "huggingface") {
                    download_success = downloadFromHuggingFace(model_id, output_dir, error_message);
                } else {
                    error_message = "Unsupported model source";
                    download_success = false;
                }

                if (!download_success) {
                    spdlog::error("Failed to download model {}: {}", model_id, error_message);

                    // Cleanup any partially downloaded files
                    try {
                        if (fs::exists(output_dir)) {
                            fs::remove_all(output_dir);
                        }
                    } catch (const fs::filesystem_error& e) {
                        spdlog::warn("Failed to clean up after failed download: {}", e.what());
                    }

                    return false;
                }

                // Perform quantization if requested
                if (quantize) {
                    // Map 'quantization_type' to 'quantization_method'
                    info.quantization_method = quantization_type;

                    if (!quantizeModel(output_dir, quantization_type, error_message)) {
                        spdlog::error("Failed to quantize model: {}", error_message);

                        // Don't delete the downloaded model, just mark quantization as failed
                        spdlog::info("Model downloaded but quantization failed");
                        info.quantization_method = ""; // Clear quantization flag
                    }
                }

                // Update model info with file metadata
                try {
                    // Check for config.json to extract more information
                    fs::path config_path = output_dir / "config.json";
                    if (fs::exists(config_path)) {
                        std::ifstream config_file(config_path);
                        json config = json::parse(config_file);

                        if (config.contains("architecture")) {
                            info.architecture_type = config["architecture"].get<std::string>();
                        } else if (config.contains("model_type")) {
                            info.architecture_type = config["model_type"].get<std::string>();
                        } else {
                            info.architecture_type = "unknown";
                        }

                        // Try to determine parameter count
                        if (config.contains("n_params")) {
                            info.parameter_count = config["n_params"].get<int64_t>();
                        } else if (config.contains("num_parameters")) {
                            info.parameter_count = config["num_parameters"].get<int64_t>();
                        } else {
                            // Make an educated guess based on model architecture
                            if (info.architecture_type == "llama" || info.architecture_type == "mistral") {
                                if (info.id.find("7b") != std::string::npos) {
                                    info.parameter_count = 7000000000;
                                } else if (info.id.find("13b") != std::string::npos) {
                                    info.parameter_count = 13000000000;
                                } else if (info.id.find("70b") != std::string::npos) {
                                    info.parameter_count = 70000000000;
                                } else {
                                    // Default to a small model size if can't determine
                                    info.parameter_count = 1000000000;
                                }
                            } else {
                                info.parameter_count = 0;
                            }
                        }
                    } else {
                        info.architecture_type = "unknown";
                        info.parameter_count = 0;
                    }

                    // Model is not fine-tuned (it's a base model)
                    info.is_finetuned = false;

                    // Add to database
                    if (!addModel(info)) {
                        spdlog::error("Failed to add model to database");
                        return false;
                    }

                } catch (const std::exception& e) {
                    spdlog::error("Error processing downloaded model: {}", e.what());
                    return false;
                }

                spdlog::info("Model {} downloaded and registered successfully", model_id);
                return true;
            }

            bool ModelManager::updateModel(const ModelInfo& info) {
                std::lock_guard<std::mutex> lock(mutex_);

                if (!available_models_.count(info.id)) {
                    spdlog::error("Cannot update model {}: not found", info.id);
                    return false;
                }

                const char* update_sql =
                    "UPDATE models SET "
                    "name = ?, source = ?, architecture = ?, parameter_count = ?, "
                    "quantization_type = ?, last_modified = ?, size_bytes = ?, "
                    "is_fine_tuned = ?, parent_model = ?, description = ?, metadata = ? "
                    "WHERE id = ?;";

                sqlite3_stmt* stmt;
                int rc = sqlite3_prepare_v2(metadata_db_, update_sql, -1, &stmt, nullptr);

                if (rc != SQLITE_OK) {
                    spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
                    return false;
                }

                sqlite3_bind_text(stmt, 1, info.name.c_str(), -1, SQLITE_STATIC);

                // Default value for 'source' since it's not in ModelInfo
                std::string source = "unknown";
                sqlite3_bind_text(stmt, 2, source.c_str(), -1, SQLITE_STATIC);

                // Map 'architecture_type' to 'architecture'
                sqlite3_bind_text(stmt, 3, info.architecture_type.c_str(), -1, SQLITE_STATIC);
                sqlite3_bind_int64(stmt, 4, info.parameter_count);

                // Map 'quantization_method' to 'quantization_type'
                if (info.quantization_method.empty()) {
                    sqlite3_bind_null(stmt, 5);
                } else {
                    sqlite3_bind_text(stmt, 5, info.quantization_method.c_str(), -1, SQLITE_STATIC);
                }

                // Update last_modified timestamp
                auto now = std::chrono::system_clock::now();
                auto now_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
                std::string current_time = ss.str();

                sqlite3_bind_text(stmt, 6, current_time.c_str(), -1, SQLITE_STATIC);

                // Calculate size_bytes
                size_t size_bytes = 0;
                fs::path model_dir = getModelPath(info.id);
                if (fs::exists(model_dir)) {
                    for (const auto& entry : fs::recursive_directory_iterator(model_dir)) {
                        if (entry.is_regular_file()) {
                            size_bytes += entry.file_size();
                        }
                    }
                }
                sqlite3_bind_int64(stmt, 7, size_bytes);

                // Map 'is_finetuned' to 'is_fine_tuned'
                sqlite3_bind_int(stmt, 8, info.is_finetuned ? 1 : 0);

                // Map 'parent_model_id' to 'parent_model'
                if (!info.parent_model_id.has_value() || info.parent_model_id->empty()) {
                    sqlite3_bind_null(stmt, 9);
                } else {
                    sqlite3_bind_text(stmt, 9, info.parent_model_id->c_str(), -1, SQLITE_STATIC);
                }

                // Default value for 'description'
                sqlite3_bind_null(stmt, 10);

                // Empty metadata as JSON string
                json empty_metadata = json::object();
                std::string metadata_json = empty_metadata.dump();
                sqlite3_bind_text(stmt, 11, metadata_json.c_str(), -1, SQLITE_STATIC);

                sqlite3_bind_text(stmt, 12, info.id.c_str(), -1, SQLITE_STATIC);

                rc = sqlite3_step(stmt);
                sqlite3_finalize(stmt);

                if (rc != SQLITE_DONE) {
                    spdlog::error("Failed to update model: {}", sqlite3_errmsg(metadata_db_));
                    return false;
                }

                // Update in-memory cache
                available_models_[info.id] = info;

                spdlog::info("Updated model {} in database", info.id);
                return true;
            }

            bool ModelManager::removeModel(const std::string& id, bool delete_files) {
                std::lock_guard<std::mutex> lock(mutex_);

                if (!available_models_.count(id)) {
                    spdlog::error("Cannot remove model {}: not found", id);
                    return false;
                }

                // Remove from database first
                const char* delete_sql = "DELETE FROM models WHERE id = ?;";
                sqlite3_stmt* stmt;
                int rc = sqlite3_prepare_v2(metadata_db_, delete_sql, -1, &stmt, nullptr);

                if (rc != SQLITE_OK) {
                    spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
                    return false;
                }

                sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);

                rc = sqlite3_step(stmt);
                sqlite3_finalize(stmt);

                if (rc != SQLITE_DONE) {
                    spdlog::error("Failed to delete model from database: {}", sqlite3_errmsg(metadata_db_));
                    return false;
                }

                // Remove model files if requested
                if (delete_files) {
                    try {
                        fs::path model_dir = getModelPath(id);

                        if (fs::exists(model_dir)) {
                            size_t removed_files = 0;
                            for (const auto& entry : fs::recursive_directory_iterator(model_dir)) {
                                if (fs::remove(entry.path())) {
                                    removed_files++;
                                }
                            }

                            fs::remove(model_dir);

                            spdlog::info("Removed {} files for model {}", removed_files, id);
                        } else {
                            spdlog::warn("Model directory for {} not found", id);
                        }
                    } catch (const fs::filesystem_error& e) {
                        spdlog::error("Error deleting model files: {}", e.what());
                        // Continue with removal from cache
                    }
                }

                // Remove from cache
                available_models_.erase(id);

                spdlog::info("Removed model {} from database", id);
                return true;
            }

            std::optional<ModelInfo> ModelManager::getModelInfo(const std::string& id) const {
                std::lock_guard<std::mutex> lock(mutex_);

                auto it = available_models_.find(id);
                if (it != available_models_.end()) {
                    return it->second;
                }

                return std::nullopt;
            }

            std::vector<ModelInfo> ModelManager::listAvailableModels() const {
                std::lock_guard<std::mutex> lock(mutex_);

                std::vector<ModelInfo> models;
                models.reserve(available_models_.size());

                for (const auto& [_, info] : available_models_) {
                    models.push_back(info);
                }

                // Sort by name
                std::sort(models.begin(), models.end(),
                          [](const ModelInfo& a, const ModelInfo& b) {
                              return a.name < b.name;
                          });

                return models;
            }

            bool ModelManager::isModelAvailable(const std::string& id) const {
                std::lock_guard<std::mutex> lock(mutex_);
                return available_models_.count(id) > 0;
            }

            fs::path ModelManager::getModelPath(const std::string& id) const {
                return models_dir_ / id;
            }

            } // namespace localllm
