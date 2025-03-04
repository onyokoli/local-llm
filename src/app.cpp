#include "app.h"
#include "model_manager.h"
#include "dataset_manager.h"
#include "fine_tuning_engine.h"
#include "inference_server.h"
#include "version.h"

#include <spdlog/spdlog.h>
#include <iostream>

namespace localllm {

LocalLLMApp::LocalLLMApp(const fs::path& base_dir, uint16_t server_port)
    : base_dir_(base_dir) {

    // Create base directory if it doesn't exist
    if (!fs::exists(base_dir_)) {
        spdlog::info("Creating base directory: {}", base_dir_.string());
        fs::create_directories(base_dir_);
    }

    // Create subdirectories
    fs::create_directories(base_dir_ / "models");
    fs::create_directories(base_dir_ / "datasets");
    fs::create_directories(base_dir_ / "logs");

    // Initialize managers
    model_manager_ = std::make_unique<ModelManager>(base_dir_ / "models");
    dataset_manager_ = std::make_unique<DatasetManager>(base_dir_ / "datasets");
    fine_tuning_engine_ = std::make_unique<FineTuningEngine>(*model_manager_, *dataset_manager_);
    inference_server_ = std::make_unique<InferenceServer>(*model_manager_, "localhost", server_port);
}

// Important: Need a defined destructor when using unique_ptr with forward-declared types
LocalLLMApp::~LocalLLMApp() = default;

void LocalLLMApp::initialize() {
    spdlog::info("Scanning available models and datasets...");
    model_manager_->scanAvailableModels();
    dataset_manager_->scanAvailableDatasets();
}

// Model Management
bool LocalLLMApp::downloadModel(const std::string& model_id, const std::string& source,
                              bool quantize, const std::string& quantization_type) {
    return model_manager_->downloadModel(model_id, source, quantize, quantization_type);
}

std::vector<ModelInfo> LocalLLMApp::listModels() {
    return model_manager_->listAvailableModels();
}

std::optional<ModelInfo> LocalLLMApp::getModelInfo(const std::string& id) {
    return model_manager_->getModelInfo(id);
}

bool LocalLLMApp::removeModel(const std::string& id, bool delete_files) {
    return model_manager_->removeModel(id, delete_files);
}

// Dataset Management
std::string LocalLLMApp::importDataset(const fs::path& path, const std::string& name,
                                    const std::string& format) {
    return dataset_manager_->importDataset(path, name, format);
}

std::vector<DatasetInfo> LocalLLMApp::listDatasets() {
    return dataset_manager_->listAvailableDatasets();
}

std::optional<DatasetInfo> LocalLLMApp::getDatasetInfo(const std::string& id) {
    return dataset_manager_->getDatasetInfo(id);
}

// Fine-tuning
bool LocalLLMApp::startFineTuning(const std::string& model_id, const std::string& dataset_id,
                                const FineTuneConfig& config) {
    return fine_tuning_engine_->startFineTuning(model_id, dataset_id, config);
}

void LocalLLMApp::stopFineTuning(bool wait_for_completion) {
    fine_tuning_engine_->stopTraining(wait_for_completion);
}

bool LocalLLMApp::isFineTuningActive() {
    return fine_tuning_engine_->isTraining();
}

// Inference
bool LocalLLMApp::startInferenceServer() {
    return inference_server_->start();
}

void LocalLLMApp::stopInferenceServer() {
    inference_server_->stop();
}

// Utility methods
std::string LocalLLMApp::getVersionInfo() {
    std::string version_info = "LocalLLM version ";
    version_info += VERSION;
    version_info += " (";
    version_info += BUILD_TYPE;
    version_info += ", built on ";
    version_info += BUILD_DATE;
    version_info += " ";
    version_info += BUILD_TIME;
    version_info += ")\n";

    version_info += "Features: ";
    version_info += "CUDA=";
    version_info += CUDA_SUPPORTED ? "yes" : "no";
    version_info += ", OpenMP=";
    version_info += OPENMP_SUPPORTED ? "yes" : "no";
    version_info += ", BLAS=";
    version_info += BLAS_SUPPORTED ? "yes" : "no";

    return version_info;
}

void LocalLLMApp::shutdown() {
    spdlog::info("Shutting down LocalLLM...");

    // Stop any active processes
    if (isFineTuningActive()) {
        spdlog::info("Stopping fine-tuning process...");
        stopFineTuning(true);
    }

    spdlog::info("Stopping inference server...");
    stopInferenceServer();

    spdlog::info("LocalLLM shutdown complete.");
}

} // namespace localllm
