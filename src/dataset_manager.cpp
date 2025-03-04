#include "dataset_manager.h"
#include <sqlite3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <regex>
#include <thread>
#include <future>
#include <atomic>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include "ggml.h"
#include "llama.h"
#include <uuid/uuid.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace localllm {

// Constants for dataset processing
constexpr size_t MAX_SAMPLE_ROWS = 1000;      // Max rows to sample for analysis
constexpr size_t MAX_TOKEN_SEQUENCE = 2048;   // Max tokens in a sequence

DatasetManager::DatasetManager(const fs::path& datasets_dir)
    : datasets_dir_(datasets_dir), metadata_db_path_(datasets_dir / "datasets_metadata.db") {

    try {
        // Create datasets directory if it doesn't exist
        if (!fs::exists(datasets_dir_)) {
            spdlog::info("Creating datasets directory at {}", datasets_dir_.string());
            fs::create_directories(datasets_dir_);
        }

        // Initialize SQLite database
        initMetadataDB();

        // Scan for existing datasets
        scanAvailableDatasets();

    } catch (const std::exception& e) {
        spdlog::error("Error initializing DatasetManager: {}", e.what());
        throw;
    }
}

DatasetManager::~DatasetManager() {
    // Close SQLite database connection
    if (metadata_db_) {
        sqlite3_close(metadata_db_);
        metadata_db_ = nullptr;
    }
}

void DatasetManager::initMetadataDB() {
    int rc = sqlite3_open(metadata_db_path_.c_str(), &metadata_db_);

    if (rc) {
        std::string error_msg = sqlite3_errmsg(metadata_db_);
        sqlite3_close(metadata_db_);
        metadata_db_ = nullptr;
        throw std::runtime_error("Cannot open database: " + error_msg);
    }

    // Create datasets table if it doesn't exist
    const char* create_table_sql =
        "CREATE TABLE IF NOT EXISTS datasets ("
        "id TEXT PRIMARY KEY,"
        "name TEXT NOT NULL,"
        "format TEXT NOT NULL,"
        "path TEXT NOT NULL,"
        "date_added TEXT NOT NULL,"
        "last_modified TEXT NOT NULL,"
        "size_bytes INTEGER NOT NULL,"
        "num_examples INTEGER NOT NULL,"
        "estimated_tokens INTEGER NOT NULL,"
        "has_instruction BOOLEAN NOT NULL,"
        "has_input BOOLEAN NOT NULL,"
        "has_output BOOLEAN NOT NULL,"
        "processed BOOLEAN NOT NULL,"
        "processed_path TEXT,"
        "sample_data TEXT,"
        "metadata TEXT"
        ");";

    char* err_msg = nullptr;
    rc = sqlite3_exec(metadata_db_, create_table_sql, nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        std::string error = err_msg;
        sqlite3_free(err_msg);
        throw std::runtime_error("SQL error: " + error);
    }

    spdlog::debug("Dataset metadata database initialized");
}

void DatasetManager::scanAvailableDatasets() {
    std::lock_guard<std::mutex> lock(mutex_);

    available_datasets_.clear();

    // First, get all datasets from database
    const char* query = "SELECT * FROM datasets;";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(metadata_db_, query, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
        return;
    }

    // Read all datasets from database and add to cache
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DatasetInfo info;

        info.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        info.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // Map 'format' to 'format_type'
        info.format_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        // Map 'path' to 'data_path'
        info.data_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        // date_added and last_modified not in DatasetInfo, read into local variables
        std::string date_added = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        std::string last_modified = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));

        // Set created_date from date_added
        info.created_date = date_added;

        // Read size_bytes but don't store it (not used in this function)
        sqlite3_column_int64(stmt, 6);

        info.num_examples = sqlite3_column_int64(stmt, 7);

        // estimated_tokens maps to total_tokens
        info.total_tokens = sqlite3_column_int64(stmt, 8);

        // Store has_instruction, has_input, has_output in format_config
        info.format_config["has_instruction"] = sqlite3_column_int(stmt, 9) != 0;
        info.format_config["has_input"] = sqlite3_column_int(stmt, 10) != 0;
        info.format_config["has_output"] = sqlite3_column_int(stmt, 11) != 0;

        // Map 'processed' to 'is_processed'
        info.is_processed = sqlite3_column_int(stmt, 12) != 0;

        // processed_path not in DatasetInfo
        if (sqlite3_column_type(stmt, 13) != SQLITE_NULL) {
            info.format_config["processed_path"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 13));
        }

        // sample_data not in DatasetInfo - can be stored in format_config
        if (sqlite3_column_type(stmt, 14) != SQLITE_NULL) {
            try {
                const char* sample_data_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 14));
                json sample_data = json::parse(sample_data_str);
                info.format_config["sample_data"] = sample_data;
            } catch (const json::exception& e) {
                spdlog::warn("Error parsing JSON sample data for dataset {}: {}", info.id, e.what());
                info.format_config["sample_data"] = json::array();
            }
        } else {
            info.format_config["sample_data"] = json::array();
        }

        // metadata not in DatasetInfo - can be merged into format_config
        if (sqlite3_column_type(stmt, 15) != SQLITE_NULL) {
            try {
                const char* metadata_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 15));
                json metadata = json::parse(metadata_str);

                // Merge metadata into format_config
                for (auto& [key, value] : metadata.items()) {
                    info.format_config[key] = value;
                }
            } catch (const json::exception& e) {
                spdlog::warn("Error parsing JSON metadata for dataset {}: {}", info.id, e.what());
            }
        }

        // Check if dataset files actually exist
        fs::path dataset_path = info.data_path;
        if (fs::exists(dataset_path)) {
            available_datasets_[info.id] = info;
        } else {
            spdlog::warn("Dataset file for {} not found at {}. Removing from database.",
                         info.id, dataset_path.string());

            // Remove from database
            std::string delete_sql = "DELETE FROM datasets WHERE id = ?;";
            sqlite3_stmt* delete_stmt;

            rc = sqlite3_prepare_v2(metadata_db_, delete_sql.c_str(), -1, &delete_stmt, nullptr);
            if (rc == SQLITE_OK) {
                sqlite3_bind_text(delete_stmt, 1, info.id.c_str(), -1, SQLITE_STATIC);
                sqlite3_step(delete_stmt);
                sqlite3_finalize(delete_stmt);
            }
        }
    }

    sqlite3_finalize(stmt);

    spdlog::info("Scan complete. Found {} datasets", available_datasets_.size());
}

std::string DatasetManager::generateUniqueId() {
    uuid_t uuid;
    uuid_generate(uuid);

    char uuid_str[37];  // 36 characters + null terminator
    uuid_unparse_lower(uuid, uuid_str);

    return std::string(uuid_str);
}

std::string DatasetManager::replaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

uint64_t DatasetManager::estimateTokenCount(const std::string& text) {
    // Simple heuristic: average English word is ~4.7 characters, and tokens are roughly word-sized
    // This is a very rough approximation; actual tokenization depends on the specific tokenizer
    constexpr double AVG_CHARS_PER_TOKEN = 4.7;
    return static_cast<uint64_t>(text.length() / AVG_CHARS_PER_TOKEN);
}

std::string DatasetManager::importDataset(const fs::path& source_path, const std::string& name,
                                        const std::string& format_type) {
    if (!fs::exists(source_path)) {
        spdlog::error("Source file does not exist: {}", source_path.string());
        throw std::runtime_error("Source file does not exist");
    }

    // Generate a unique ID for the dataset
    std::string dataset_id = generateUniqueId();

    // Create dataset info
    DatasetInfo info;
    info.id = dataset_id;
    info.name = name.empty() ? source_path.filename().string() : name;

    // Map 'format' to 'format_type'
    info.format_type = format_type;

    // Copy the dataset to our datasets directory
    fs::path destination = datasets_dir_ / (dataset_id + source_path.extension().string());

    try {
        fs::copy_file(source_path, destination, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& e) {
        spdlog::error("Failed to copy dataset file: {}", e.what());
        throw std::runtime_error("Failed to copy dataset file: " + std::string(e.what()));
    }

    // Map 'path' to 'data_path'
    info.data_path = destination;

    // Set timestamps
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");

    std::string date_added = ss.str();

    // Set created_date
    info.created_date = date_added;

    // Initialize other fields
    info.is_processed = false;
    info.num_examples = 0;
    info.total_tokens = 0;

    // Initialize format_config with default values
    info.format_config["has_instruction"] = false;
    info.format_config["has_input"] = false;
    info.format_config["has_output"] = false;

    // Analyze the dataset
    spdlog::info("Analyzing dataset...");
    if (!analyzeDataset(info)) {
        spdlog::error("Failed to analyze dataset");

        // Clean up the copied file
        try {
            fs::remove(destination);
        } catch (...) {
            // Ignore cleanup errors
        }

        throw std::runtime_error("Failed to analyze dataset");
    }

    // Add to database
    if (!addDatasetToDb(info)) {
        spdlog::error("Failed to add dataset to database");

        // Clean up the copied file
        try {
            fs::remove(destination);
        } catch (...) {
            // Ignore cleanup errors
        }

        throw std::runtime_error("Failed to add dataset to database");
    }

    // Add to cache
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_datasets_[dataset_id] = info;
    }

    spdlog::info("Dataset imported successfully: {} ({} examples, ~{} tokens)",
                 info.name, info.num_examples, info.total_tokens);

    return dataset_id;
}

bool DatasetManager::analyzeDataset(DatasetInfo& info) {
    spdlog::info("Analyzing {} dataset: {}", info.format_type, info.data_path.string());

    if (info.format_type == "jsonl") {
        return analyzeJsonlDataset(info);
    } else if (info.format_type == "text" || info.format_type == "txt") {
        return analyzeTextDataset(info);
    } else if (info.format_type == "csv") {
        return analyzeCsvDataset(info);
    } else {
        spdlog::error("Unsupported dataset format: {}", info.format_type);
        return false;
    }
}

bool DatasetManager::analyzeJsonlDataset(DatasetInfo& info) {
    std::ifstream file(info.data_path);
    if (!file.is_open()) {
        spdlog::error("Failed to open dataset file: {}", info.data_path.string());
        return false;
    }

    std::string line;
    size_t line_count = 0;
    uint64_t total_tokens = 0;
    bool has_instruction = false;
    bool has_input = false;
    bool has_output = false;
    std::vector<json> samples;

    // Process each line
    while (std::getline(file, line) && line_count < MAX_SAMPLE_ROWS) {
        if (line.empty()) continue;

        try {
            json example = json::parse(line);

            // Determine dataset structure
            if (example.contains("instruction")) {
                has_instruction = true;
                total_tokens += estimateTokenCount(example["instruction"].get<std::string>());
            }

            if (example.contains("input")) {
                has_input = true;
                if (!example["input"].is_null()) {
                    total_tokens += estimateTokenCount(example["input"].get<std::string>());
                }
            }

            if (example.contains("output")) {
                has_output = true;
                total_tokens += estimateTokenCount(example["output"].get<std::string>());
            }

            // Check if it's a conversation format
            if (example.contains("messages") && example["messages"].is_array()) {
                for (const auto& message : example["messages"]) {
                    if (message.contains("content")) {
                        total_tokens += estimateTokenCount(message["content"].get<std::string>());
                    }
                }
            }

            // Store samples for analysis
            if (samples.size() < 5) {
                samples.push_back(example);
            }

            line_count++;
        } catch (const json::exception& e) {
            spdlog::warn("Error parsing JSON at line {}: {}", line_count + 1, e.what());
            continue;
        }
    }

    // Count total lines in file for larger datasets
    if (file.eof()) {
        info.num_examples = line_count;
    } else {
        // We've only read a sample, so estimate total count based on file size
        const size_t current_position = file.tellg();
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        info.num_examples = static_cast<uint64_t>(line_count * (static_cast<double>(file_size) / current_position));
    }

    // If we read a sample, extrapolate token count
    if (line_count < info.num_examples) {
        total_tokens = static_cast<uint64_t>(total_tokens * (static_cast<double>(info.num_examples) / line_count));
    }

    info.total_tokens = total_tokens;

    // Store these flags in format_config for later use
    info.format_config["has_instruction"] = has_instruction;
    info.format_config["has_input"] = has_input;
    info.format_config["has_output"] = has_output;
    info.format_config["sample_data"] = samples;

    // Add metadata about structure
    info.format_config["structure"] = {
        {"has_instruction", has_instruction},
        {"has_input", has_input},
        {"has_output", has_output}
    };

    return true;
}

bool DatasetManager::analyzeTextDataset(DatasetInfo& info) {
    std::ifstream file(info.data_path);
    if (!file.is_open()) {
        spdlog::error("Failed to open dataset file: {}", info.data_path.string());
        return false;
    }

    // For text files, we'll treat each line as a separate example
    // This is a simple approach; you might want more sophisticated parsing

    std::string line;
    size_t line_count = 0;
    uint64_t total_tokens = 0;
    std::vector<json> samples;

    // Process each line
    while (std::getline(file, line) && line_count < MAX_SAMPLE_ROWS) {
        if (line.empty()) continue;

        total_tokens += estimateTokenCount(line);

        // Store samples for analysis
        if (samples.size() < 5) {
            samples.push_back({{"text", line}});
        }

        line_count++;
    }

    // Count total lines in file for larger datasets
    if (file.eof()) {
        info.num_examples = line_count;
    } else {
        // We've only read a sample, so estimate total count based on file size
        const size_t current_position = file.tellg();
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        info.num_examples = static_cast<uint64_t>(line_count * (static_cast<double>(file_size) / current_position));
    }

    // If we read a sample, extrapolate token count
    if (line_count < info.num_examples) {
        total_tokens = static_cast<uint64_t>(total_tokens * (static_cast<double>(info.num_examples) / line_count));
    }

    info.total_tokens = total_tokens;

    // Store these flags in format_config for later use
    info.format_config["has_instruction"] = false;
    info.format_config["has_input"] = true;  // We treat the text as input
    info.format_config["has_output"] = false;
    info.format_config["sample_data"] = samples;

    // Add metadata about structure
    info.format_config["structure"] = {
        {"has_instruction", false},
        {"has_input", true},
        {"has_output", false},
        {"format", "text_line"}
    };

    return true;
}

bool DatasetManager::analyzeCsvDataset(DatasetInfo& info) {
    std::ifstream file(info.data_path);
    if (!file.is_open()) {
        spdlog::error("Failed to open dataset file: {}", info.data_path.string());
        return false;
    }

    std::string line;
    size_t line_count = 0;
    uint64_t total_tokens = 0;
    std::vector<json> samples;
    std::vector<std::string> headers;

    // First line should be headers
    if (std::getline(file, line)) {
        // Parse CSV header
        std::istringstream header_stream(line);
        std::string header;
        while (std::getline(header_stream, header, ',')) {
            // Remove quotes if present
            if (header.size() >= 2 && header.front() == '"' && header.back() == '"') {
                header = header.substr(1, header.size() - 2);
            }
            headers.push_back(header);
        }
    }

    // Process data rows
    bool has_instruction = false;
    bool has_input = false;
    bool has_output = false;

    // Check headers for common fields
    for (const auto& header : headers) {
        if (header == "instruction" || header == "prompt") {
            has_instruction = true;
        } else if (header == "input" || header == "context") {
            has_input = true;
        } else if (header == "output" || header == "response" || header == "completion") {
            has_output = true;
        }
    }

    // Process each data row
    while (std::getline(file, line) && line_count < MAX_SAMPLE_ROWS) {
        if (line.empty()) continue;

        std::istringstream row_stream(line);
        std::string field;
        size_t field_index = 0;
        json row_obj;

        // Parse CSV row
        while (std::getline(row_stream, field, ',')) {
            // Handle quoted fields
            if (field.size() >= 2 && field.front() == '"' && field.back() == '"') {
                field = field.substr(1, field.size() - 2);
            }

            // Map to header if available
            if (field_index < headers.size()) {
                row_obj[headers[field_index]] = field;
                total_tokens += estimateTokenCount(field);
            }

            field_index++;
        }

        // Store samples for analysis
        if (samples.size() < 5) {
            samples.push_back(row_obj);
        }

        line_count++;
    }

    // Count total lines in file for larger datasets
    if (file.eof()) {
        info.num_examples = line_count;
    } else {
        // We've only read a sample, so estimate total count based on file size
        const size_t current_position = file.tellg();
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        info.num_examples = static_cast<uint64_t>(line_count * (static_cast<double>(file_size) / current_position));
    }

    // If we read a sample, extrapolate token count
    if (line_count < info.num_examples) {
        total_tokens = static_cast<uint64_t>(total_tokens * (static_cast<double>(info.num_examples) / line_count));
    }

    info.total_tokens = total_tokens;

    // Store these flags in format_config for later use
    info.format_config["has_instruction"] = has_instruction;
    info.format_config["has_input"] = has_input;
    info.format_config["has_output"] = has_output;
    info.format_config["sample_data"] = samples;

    // Add metadata about structure
    info.format_config["structure"] = {
        {"has_instruction", has_instruction},
        {"has_input", has_input},
        {"has_output", has_output},
        {"headers", headers}
    };

    return true;
}

bool DatasetManager::preprocessDataset(const std::string& id, llama_context* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if dataset exists
    auto it = available_datasets_.find(id);
    if (it == available_datasets_.end()) {
        spdlog::error("Dataset not found: {}", id);
        return false;
    }

    DatasetInfo& info = it->second;

    // Create directory for processed files
    fs::path processed_dir = datasets_dir_ / (id + "_processed");
    if (!fs::exists(processed_dir)) {
        fs::create_directories(processed_dir);
    }

    // Process dataset based on format
    bool success = false;
    if (info.format_type == "jsonl") {
        success = preprocessJsonlDataset(info, processed_dir, ctx);
    } else if (info.format_type == "text" || info.format_type == "txt") {
        success = preprocessTextDataset(info, processed_dir, ctx);
    } else if (info.format_type == "csv") {
        success = preprocessCsvDataset(info, processed_dir, ctx);
    } else {
        spdlog::error("Unsupported dataset format for preprocessing: {}", info.format_type);
        return false;
    }

    if (success) {
        // Update dataset info
        info.is_processed = true;

        // Store processed_path in format_config
        info.format_config["processed_path"] = processed_dir.string();

        // Update last_modified timestamp
        std::string last_modified = [&]() {
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
            return ss.str();
        }();

        // Update in database
        if (!updateDatasetInDb(info)) {
            spdlog::error("Failed to update dataset metadata after preprocessing");
            return false;
        }

        spdlog::info("Dataset preprocessing completed successfully: {}", id);
        return true;
    } else {
        spdlog::error("Dataset preprocessing failed: {}", id);
        return false;
    }
}

std::vector<llama_token> DatasetManager::tokenize(llama_context* ctx, const std::string& text, bool add_bos) {
    // Fix the min function to use the same type - convert MAX_TOKEN_SEQUENCE to int
    const int n_ctx = llama_n_ctx(ctx);
    const int max_tokens = std::min(n_ctx, static_cast<int>(MAX_TOKEN_SEQUENCE));

    std::vector<llama_token> result(max_tokens);

    // Get the vocabulary from the context - needed for newer llama.cpp API
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_get_vocab(model);

    // Get the BOS token ID
    const llama_token bos_id = llama_vocab_bos(vocab);

    int n = 0;
    if (add_bos) {
        // Add BOS token manually
        result[0] = bos_id;

        // Use llama_tokenize with all 7 parameters
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                          result.data() + 1, max_tokens - 1, false, true);

        if (n >= 0) {
            n += 1;  // Account for the BOS token
        } else {
            n = -n + 1;  // For truncated results
        }
    } else {
        // Use llama_tokenize with all 7 parameters
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                          result.data(), max_tokens, false, true);
    }

    if (n < 0) {
        n = -n;  // If truncated, llama_tokenize returns negative count
    }

    result.resize(n);
    return result;
}

void DatasetManager::writeTokenizedExample(std::ofstream& file, const std::vector<llama_token>& input_tokens,
                                         const std::vector<llama_token>& output_tokens) {
    // Format: <input tokens length><output tokens length><input tokens><output tokens>
    // First write the lengths as 4-byte integers
    uint32_t input_size = static_cast<uint32_t>(input_tokens.size());
    uint32_t output_size = static_cast<uint32_t>(output_tokens.size());

    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    // Then write the tokens as int32 values
    if (!input_tokens.empty()) {
        file.write(reinterpret_cast<const char*>(input_tokens.data()),
                   input_tokens.size() * sizeof(llama_token));
    }

    if (!output_tokens.empty()) {
        file.write(reinterpret_cast<const char*>(output_tokens.data()),
                   output_tokens.size() * sizeof(llama_token));
    }
}

bool DatasetManager::preprocessJsonlDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx) {
    spdlog::info("Preprocessing JSONL dataset: {}", info.data_path.string());

    std::ifstream infile(info.data_path);
    if (!infile.is_open()) {
        spdlog::error("Failed to open dataset file: {}", info.data_path.string());
        return false;
    }

    // Open output binary file for tokenized data
    fs::path outfile_path = processed_dir / "tokenized.bin";
    std::ofstream outfile(outfile_path, std::ios::binary);
    if (!outfile.is_open()) {
        spdlog::error("Failed to create output file: {}", outfile_path.string());
        return false;
    }

    std::string line;
    size_t processed_count = 0;
    size_t error_count = 0;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        try {
            json example = json::parse(line);

            std::string instruction, input, output;

            // Handle different dataset formats
            if (example.contains("instruction") && example.contains("output")) {
                // Standard instruction format
                instruction = example["instruction"].get<std::string>();

                if (example.contains("input") && !example["input"].is_null()) {
                    input = example["input"].get<std::string>();
                }

                output = example["output"].get<std::string>();

                // Format prompt based on whether input is present
                std::string prompt;
                if (!input.empty()) {
                    prompt = "Instruction: " + instruction + "\n\nInput: " + input + "\n\nOutput: ";
                } else {
                    prompt = "Instruction: " + instruction + "\n\nOutput: ";
                                    }

                                    // Tokenize prompt and response
                                    auto input_tokens = tokenize(ctx, prompt, true);  // Add BOS token
                                    auto output_tokens = tokenize(ctx, output, false);  // No BOS token for output

                                    // Write to binary file
                                    writeTokenizedExample(outfile, input_tokens, output_tokens);

                                } else if (example.contains("messages") && example["messages"].is_array()) {
                                    // Chat format
                                    std::string prompt = "";
                                    std::string output = "";

                                    for (size_t i = 0; i < example["messages"].size(); i++) {
                                        const auto& message = example["messages"][i];

                                        if (!message.contains("role") || !message.contains("content")) {
                                            continue;
                                        }

                                        std::string role = message["role"].get<std::string>();
                                        std::string content = message["content"].get<std::string>();

                                        if (i < example["messages"].size() - 1) {
                                            // Messages before the last one go into the prompt
                                            prompt += "<" + role + ">\n" + content + "\n";
                                        } else {
                                            // Last message is the expected output (usually assistant's response)
                                            if (role == "assistant") {
                                                prompt += "<assistant>\n";
                                                output = content;
                                            } else {
                                                // If last message isn't from assistant, include it in prompt
                                                prompt += "<" + role + ">\n" + content + "\n<assistant>\n";
                                                output = "";  // No expected output
                                            }
                                        }
                                    }

                                    if (!output.empty()) {
                                        // Tokenize prompt and response
                                        auto input_tokens = tokenize(ctx, prompt, true);  // Add BOS token
                                        auto output_tokens = tokenize(ctx, output, false);  // No BOS token for output

                                        // Write to binary file
                                        writeTokenizedExample(outfile, input_tokens, output_tokens);
                                    }

                                } else {
                                    // Unknown format, try to infer structure
                                    spdlog::warn("Unknown example format at line {}, skipping", processed_count + 1);
                                    error_count++;
                                    continue;
                                }

                                processed_count++;

                                // Progress reporting
                                if (processed_count % 1000 == 0) {
                                    spdlog::info("Processed {} examples", processed_count);
                                }
                            } catch (const json::exception& e) {
                                spdlog::warn("Error parsing JSON at line {}: {}", processed_count + 1, e.what());
                                error_count++;
                                continue;
                            }
                        }

                        outfile.close();

                        // Save preprocessing metadata
                        fs::path metadata_path = processed_dir / "metadata.json";
                        json metadata = {
                            {"format", "jsonl"},
                            {"total_examples", processed_count},
                            {"errors", error_count},
                            {"tokenizer", "llama"},
                            {"date_processed", ""} // Use current date in caller
                        };

                        std::ofstream metadata_file(metadata_path);
                        metadata_file << metadata.dump(2);
                        metadata_file.close();

                        spdlog::info("JSONL preprocessing complete: {} examples processed, {} errors",
                                     processed_count, error_count);

                        return true;
                    }

                    bool DatasetManager::preprocessTextDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx) {
                        spdlog::info("Preprocessing text dataset: {}", info.data_path.string());

                        std::ifstream infile(info.data_path);
                        if (!infile.is_open()) {
                            spdlog::error("Failed to open dataset file: {}", info.data_path.string());
                            return false;
                        }

                        // Open output binary file for tokenized data
                        fs::path outfile_path = processed_dir / "tokenized.bin";
                        std::ofstream outfile(outfile_path, std::ios::binary);
                        if (!outfile.is_open()) {
                            spdlog::error("Failed to create output file: {}", outfile_path.string());
                            return false;
                        }

                        std::string line;
                        size_t processed_count = 0;

                        // For text files, we'll use a sliding window approach for next-token prediction
                        const size_t window_size = 512;  // Tokens per window
                        const size_t stride = 256;       // Stride between windows

                        std::vector<llama_token> all_tokens;

                        // First, tokenize the entire file (or a large chunk of it)
                        std::string file_content;
                        size_t max_size = 10 * 1024 * 1024;  // 10MB max to avoid memory issues

                        // Read file content
                        infile.seekg(0, std::ios::end);
                        size_t file_size = infile.tellg();
                        infile.seekg(0, std::ios::beg);

                        if (file_size > max_size) {
                            spdlog::warn("File is large ({}MB), reading only first {}MB",
                                         file_size / (1024 * 1024), max_size / (1024 * 1024));
                            file_content.resize(max_size);
                            infile.read(&file_content[0], max_size);
                        } else {
                            file_content.resize(file_size);
                            infile.read(&file_content[0], file_size);
                        }

                        // Tokenize the content
                        all_tokens = tokenize(ctx, file_content, true);  // Add BOS token

                        // Process in sliding windows
                        for (size_t i = 0; i + window_size <= all_tokens.size(); i += stride) {
                            size_t end_idx = std::min(i + window_size, all_tokens.size());
                            size_t input_size = end_idx - i - 1;  // Last token is the target

                            if (input_size < 8) continue;  // Skip windows that are too small

                            std::vector<llama_token> input_tokens(all_tokens.begin() + i, all_tokens.begin() + i + input_size);
                            std::vector<llama_token> output_tokens = {all_tokens[i + input_size]};  // Next token is output

                            // Write to binary file
                            writeTokenizedExample(outfile, input_tokens, output_tokens);

                            processed_count++;

                            // Progress reporting
                            if (processed_count % 1000 == 0) {
                                spdlog::info("Processed {} windows", processed_count);
                            }
                        }

                        outfile.close();

                        // Save preprocessing metadata
                        fs::path metadata_path = processed_dir / "metadata.json";
                        json metadata = {
                            {"format", "text"},
                            {"total_windows", processed_count},
                            {"window_size", window_size},
                            {"stride", stride},
                            {"tokenizer", "llama"},
                            {"date_processed", ""} // Use current date in caller
                        };

                        std::ofstream metadata_file(metadata_path);
                        metadata_file << metadata.dump(2);
                        metadata_file.close();

                        spdlog::info("Text preprocessing complete: {} windows processed", processed_count);

                        return true;
                    }

                    bool DatasetManager::preprocessCsvDataset(DatasetInfo& info, const fs::path& processed_dir, llama_context* ctx) {
                        spdlog::info("Preprocessing CSV dataset: {}", info.data_path.string());

                        std::ifstream infile(info.data_path);
                        if (!infile.is_open()) {
                            spdlog::error("Failed to open dataset file: {}", info.data_path.string());
                            return false;
                        }

                        // Open output binary file for tokenized data
                        fs::path outfile_path = processed_dir / "tokenized.bin";
                        std::ofstream outfile(outfile_path, std::ios::binary);
                        if (!outfile.is_open()) {
                            spdlog::error("Failed to create output file: {}", outfile_path.string());
                            return false;
                        }

                        std::string line;
                        size_t processed_count = 0;
                        size_t error_count = 0;

                        // Parse CSV header
                        std::vector<std::string> headers;
                        if (std::getline(infile, line)) {
                            std::istringstream header_stream(line);
                            std::string header;
                            while (std::getline(header_stream, header, ',')) {
                                // Remove quotes if present
                                if (header.size() >= 2 && header.front() == '"' && header.back() == '"') {
                                    header = header.substr(1, header.size() - 2);
                                }
                                headers.push_back(header);
                            }
                        }

                        // Find important columns
                        int instruction_idx = -1;
                        int input_idx = -1;
                        int output_idx = -1;
                        int prompt_idx = -1;
                        int completion_idx = -1;

                        for (size_t i = 0; i < headers.size(); i++) {
                            std::string lower_header = headers[i];
                            std::transform(lower_header.begin(), lower_header.end(), lower_header.begin(), ::tolower);

                            if (lower_header == "instruction" || lower_header == "instructions") {
                                instruction_idx = i;
                            } else if (lower_header == "input" || lower_header == "context") {
                                input_idx = i;
                            } else if (lower_header == "output" || lower_header == "response" || lower_header == "answer") {
                                output_idx = i;
                            } else if (lower_header == "prompt" || lower_header == "question") {
                                prompt_idx = i;
                            } else if (lower_header == "completion" || lower_header == "response") {
                                completion_idx = i;
                            }
                        }

                        // Check if we found usable columns
                        bool use_instruction_format = (instruction_idx >= 0 && output_idx >= 0);
                        bool use_prompt_completion = (prompt_idx >= 0 && completion_idx >= 0);

                        if (!use_instruction_format && !use_prompt_completion) {
                            spdlog::error("Could not identify usable columns in CSV. Need either instruction+output or prompt+completion columns.");
                            return false;
                        }

                        // Process each data row
                        while (std::getline(infile, line)) {
                            if (line.empty()) continue;

                            try {
                                // Parse CSV row, handling quoted fields
                                std::vector<std::string> fields;
                                size_t start = 0;
                                bool in_quotes = false;
                                std::string field;

                                for (size_t i = 0; i <= line.length(); i++) {
                                    if (i == line.length() || (line[i] == ',' && !in_quotes)) {
                                        field = line.substr(start, i - start);

                                        // Remove quotes if present
                                        if (field.size() >= 2 && field.front() == '"' && field.back() == '"') {
                                            field = field.substr(1, field.size() - 2);
                                        }

                                        fields.push_back(field);
                                        start = i + 1;
                                    } else if (line[i] == '"') {
                                        in_quotes = !in_quotes;
                                    }
                                }

                                std::string input_text, output_text;

                                if (use_instruction_format && instruction_idx < static_cast<int>(fields.size()) &&
                                    output_idx < static_cast<int>(fields.size())) {
                                    // Process instruction-output format
                                    std::string instruction = fields[instruction_idx];
                                    std::string output = fields[output_idx];
                                    std::string input = (input_idx >= 0 && input_idx < static_cast<int>(fields.size()))
                                                       ? fields[input_idx] : "";

                                    // Format prompt based on whether input is present
                                    if (!input.empty()) {
                                        input_text = "Instruction: " + instruction + "\n\nInput: " + input + "\n\nOutput: ";
                                    } else {
                                        input_text = "Instruction: " + instruction + "\n\nOutput: ";
                                    }

                                    output_text = output;

                                } else if (use_prompt_completion && prompt_idx < static_cast<int>(fields.size()) &&
                                           completion_idx < static_cast<int>(fields.size())) {
                                    // Process prompt-completion format
                                    input_text = fields[prompt_idx];
                                    output_text = fields[completion_idx];
                                } else {
                                    // Skip rows that don't have the required fields
                                    error_count++;
                                    continue;
                                }

                                // Tokenize prompt and response
                                auto input_tokens = tokenize(ctx, input_text, true);  // Add BOS token
                                auto output_tokens = tokenize(ctx, output_text, false);  // No BOS token for output

                                // Write to binary file
                                writeTokenizedExample(outfile, input_tokens, output_tokens);

                                processed_count++;

                                // Progress reporting
                                if (processed_count % 1000 == 0) {
                                    spdlog::info("Processed {} examples", processed_count);
                                }

                            } catch (const std::exception& e) {
                                spdlog::warn("Error processing CSV row: {}", e.what());
                                error_count++;
                                continue;
                            }
                        }

                        outfile.close();

                        // Save preprocessing metadata
                        fs::path metadata_path = processed_dir / "metadata.json";
                        json metadata = {
                            {"format", "csv"},
                            {"total_examples", processed_count},
                            {"errors", error_count},
                            {"headers", headers},
                            {"tokenizer", "llama"},
                            {"date_processed", ""} // Use current date in caller
                        };

                        std::ofstream metadata_file(metadata_path);
                        metadata_file << metadata.dump(2);
                        metadata_file.close();

                        spdlog::info("CSV preprocessing complete: {} examples processed, {} errors",
                                     processed_count, error_count);

                        return true;
                    }

                    bool DatasetManager::addDatasetToDb(const DatasetInfo& info) {
                        const char* insert_sql =
                            "INSERT INTO datasets (id, name, format, path, date_added, last_modified, size_bytes, "
                            "num_examples, estimated_tokens, has_instruction, has_input, has_output, processed, "
                            "processed_path, sample_data, metadata) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

                        sqlite3_stmt* stmt;
                        int rc = sqlite3_prepare_v2(metadata_db_, insert_sql, -1, &stmt, nullptr);

                        if (rc != SQLITE_OK) {
                            spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
                            return false;
                        }

                        sqlite3_bind_text(stmt, 1, info.id.c_str(), -1, SQLITE_STATIC);
                        sqlite3_bind_text(stmt, 2, info.name.c_str(), -1, SQLITE_STATIC);

                        // Map 'format_type' to 'format'
                        sqlite3_bind_text(stmt, 3, info.format_type.c_str(), -1, SQLITE_STATIC);

                        // Map 'data_path' to 'path'
                        sqlite3_bind_text(stmt, 4, info.data_path.string().c_str(), -1, SQLITE_STATIC);

                        // Set date_added from created_date
                        sqlite3_bind_text(stmt, 5, info.created_date.c_str(), -1, SQLITE_STATIC);

                        // Set last_modified to current time
                        auto now = std::chrono::system_clock::now();
                        auto now_time_t = std::chrono::system_clock::to_time_t(now);
                        std::stringstream ss;
                        ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
                        std::string current_time = ss.str();
                        sqlite3_bind_text(stmt, 6, current_time.c_str(), -1, SQLITE_STATIC);

                        // Calculate size_bytes
                        size_t size_bytes = 0;
                        if (fs::exists(info.data_path)) {
                            size_bytes = fs::file_size(info.data_path);
                        }
                        sqlite3_bind_int64(stmt, 7, size_bytes);

                        sqlite3_bind_int64(stmt, 8, info.num_examples);

                        // Map 'total_tokens' to 'estimated_tokens'
                        sqlite3_bind_int64(stmt, 9, info.total_tokens);

                        // Get has_instruction, has_input, has_output from format_config
                        bool has_instruction = info.format_config.contains("has_instruction") ?
                                              info.format_config["has_instruction"].get<bool>() : false;
                        bool has_input = info.format_config.contains("has_input") ?
                                        info.format_config["has_input"].get<bool>() : true;
                        bool has_output = info.format_config.contains("has_output") ?
                                         info.format_config["has_output"].get<bool>() : false;

                        sqlite3_bind_int(stmt, 10, has_instruction ? 1 : 0);
                        sqlite3_bind_int(stmt, 11, has_input ? 1 : 0);
                        sqlite3_bind_int(stmt, 12, has_output ? 1 : 0);

                        // Map 'is_processed' to 'processed'
                        sqlite3_bind_int(stmt, 13, info.is_processed ? 1 : 0);

                        // Get processed_path from format_config
                        std::string processed_path = info.format_config.contains("processed_path") ?
                                                    info.format_config["processed_path"].get<std::string>() : "";

                        if (processed_path.empty()) {
                            sqlite3_bind_null(stmt, 14);
                        } else {
                            sqlite3_bind_text(stmt, 14, processed_path.c_str(), -1, SQLITE_STATIC);
                        }

                        // Get sample_data from format_config
                        json sample_data = info.format_config.contains("sample_data") ?
                                          info.format_config["sample_data"] : json::array();
                        std::string sample_data_json = sample_data.dump();
                        sqlite3_bind_text(stmt, 15, sample_data_json.c_str(), -1, SQLITE_STATIC);

                        // Store other metadata from format_config
                        json metadata = info.format_config;

                        // Remove keys already stored elsewhere to avoid duplication
                        if (metadata.contains("sample_data")) metadata.erase("sample_data");
                        if (metadata.contains("processed_path")) metadata.erase("processed_path");
                        if (metadata.contains("has_instruction")) metadata.erase("has_instruction");
                        if (metadata.contains("has_input")) metadata.erase("has_input");
                        if (metadata.contains("has_output")) metadata.erase("has_output");

                        std::string metadata_json = metadata.dump();
                        sqlite3_bind_text(stmt, 16, metadata_json.c_str(), -1, SQLITE_STATIC);

                        rc = sqlite3_step(stmt);
                        sqlite3_finalize(stmt);

                        if (rc != SQLITE_DONE) {
                            spdlog::error("Failed to insert dataset: {}", sqlite3_errmsg(metadata_db_));
                            return false;
                        }

                        return true;
                    }

                    bool DatasetManager::updateDatasetInDb(const DatasetInfo& info) {
                        const char* update_sql =
                            "UPDATE datasets SET "
                            "name = ?, format = ?, path = ?, last_modified = ?, size_bytes = ?, "
                            "num_examples = ?, estimated_tokens = ?, has_instruction = ?, has_input = ?, "
                            "has_output = ?, processed = ?, processed_path = ?, sample_data = ?, metadata = ? "
                            "WHERE id = ?;";

                        sqlite3_stmt* stmt;
                        int rc = sqlite3_prepare_v2(metadata_db_, update_sql, -1, &stmt, nullptr);

                        if (rc != SQLITE_OK) {
                            spdlog::error("Failed to prepare statement: {}", sqlite3_errmsg(metadata_db_));
                            return false;
                        }

                        sqlite3_bind_text(stmt, 1, info.name.c_str(), -1, SQLITE_STATIC);

                        // Map 'format_type' to 'format'
                        sqlite3_bind_text(stmt, 2, info.format_type.c_str(), -1, SQLITE_STATIC);

                        // Map 'data_path' to 'path'
                        sqlite3_bind_text(stmt, 3, info.data_path.string().c_str(), -1, SQLITE_STATIC);

                        // Set last_modified to current time
                        auto now = std::chrono::system_clock::now();
                        auto now_time_t = std::chrono::system_clock::to_time_t(now);
                        std::stringstream ss;
                        ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
                        std::string current_time = ss.str();
                        sqlite3_bind_text(stmt, 4, current_time.c_str(), -1, SQLITE_STATIC);

                        // Calculate size_bytes
                        size_t size_bytes = 0;
                        if (fs::exists(info.data_path)) {
                            size_bytes = fs::file_size(info.data_path);
                        }
                        sqlite3_bind_int64(stmt, 5, size_bytes);

                        sqlite3_bind_int64(stmt, 6, info.num_examples);

                        // Map 'total_tokens' to 'estimated_tokens'
                        sqlite3_bind_int64(stmt, 7, info.total_tokens);

                        // Get has_instruction, has_input, has_output from format_config
                        bool has_instruction = info.format_config.contains("has_instruction") ?
                                              info.format_config["has_instruction"].get<bool>() : false;
                        bool has_input = info.format_config.contains("has_input") ?
                                        info.format_config["has_input"].get<bool>() : true;
                        bool has_output = info.format_config.contains("has_output") ?
                                         info.format_config["has_output"].get<bool>() : false;

                        sqlite3_bind_int(stmt, 8, has_instruction ? 1 : 0);
                        sqlite3_bind_int(stmt, 9, has_input ? 1 : 0);
                        sqlite3_bind_int(stmt, 10, has_output ? 1 : 0);

                        // Map 'is_processed' to 'processed'
                        sqlite3_bind_int(stmt, 11, info.is_processed ? 1 : 0);

                        // Get processed_path from format_config
                        std::string processed_path = info.format_config.contains("processed_path") ?
                                                    info.format_config["processed_path"].get<std::string>() : "";

                        if (processed_path.empty()) {
                            sqlite3_bind_null(stmt, 12);
                        } else {
                            sqlite3_bind_text(stmt, 12, processed_path.c_str(), -1, SQLITE_STATIC);
                        }

                        // Get sample_data from format_config
                        json sample_data = info.format_config.contains("sample_data") ?
                                          info.format_config["sample_data"] : json::array();
                        std::string sample_data_json = sample_data.dump();
                        sqlite3_bind_text(stmt, 13, sample_data_json.c_str(), -1, SQLITE_STATIC);

                        // Store other metadata from format_config
                        json metadata = info.format_config;

                        // Remove keys already stored elsewhere to avoid duplication
                        if (metadata.contains("sample_data")) metadata.erase("sample_data");
                        if (metadata.contains("processed_path")) metadata.erase("processed_path");
                        if (metadata.contains("has_instruction")) metadata.erase("has_instruction");
                        if (metadata.contains("has_input")) metadata.erase("has_input");
                        if (metadata.contains("has_output")) metadata.erase("has_output");

                        std::string metadata_json = metadata.dump();
                        sqlite3_bind_text(stmt, 14, metadata_json.c_str(), -1, SQLITE_STATIC);

                        sqlite3_bind_text(stmt, 15, info.id.c_str(), -1, SQLITE_STATIC);

                        rc = sqlite3_step(stmt);
                        sqlite3_finalize(stmt);

                        if (rc != SQLITE_DONE) {
                            spdlog::error("Failed to update dataset: {}", sqlite3_errmsg(metadata_db_));
                            return false;
                        }

                        return true;
                    }

                    std::optional<DatasetInfo> DatasetManager::getDatasetInfo(const std::string& id) const {
                        std::lock_guard<std::mutex> lock(mutex_);

                        auto it = available_datasets_.find(id);
                        if (it != available_datasets_.end()) {
                            return it->second;
                        }

                        return std::nullopt;
                    }

                    std::vector<DatasetInfo> DatasetManager::listAvailableDatasets() const {
                        std::lock_guard<std::mutex> lock(mutex_);

                        std::vector<DatasetInfo> datasets;
                        datasets.reserve(available_datasets_.size());

                        for (const auto& [_, info] : available_datasets_) {
                            datasets.push_back(info);
                        }

                        // Sort by name
                        std::sort(datasets.begin(), datasets.end(),
                                  [](const DatasetInfo& a, const DatasetInfo& b) {
                                      return a.name < b.name;
                                  });

                        return datasets;
                    }

                    bool DatasetManager::isDatasetAvailable(const std::string& id) const {
                        std::lock_guard<std::mutex> lock(mutex_);
                        return available_datasets_.count(id) > 0;
                    }

                    } // namespace localllm
