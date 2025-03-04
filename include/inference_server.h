#ifndef LOCALLLM_INFERENCE_SERVER_H
#define LOCALLLM_INFERENCE_SERVER_H

#include <string>
#include <unordered_map>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <optional>
#include <chrono>

#include "types.h"

// Forward declarations for httplib
namespace httplib {
class Server;
}

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;

namespace localllm {

// Forward declarations
class ModelManager;

// Inference request structure
struct InferenceRequest {
    std::string id;
    std::string model_id;
    std::string prompt;
    InferenceConfig config;
    std::chrono::steady_clock::time_point created_at;
    std::function<void(const std::string&, bool)> stream_callback;
    std::function<void(const GenerationResult&)> completion_callback;
};

// Inference server status
struct ServerStatus {
    bool running = false;
    int active_requests = 0;
    int pending_requests = 0;
    int completed_requests = 0;
    int port = 0;
    std::string host = "";
    std::vector<std::string> loaded_models;
    size_t memory_usage = 0;
    std::chrono::steady_clock::time_point start_time;
};

// Class responsible for serving models for inference
class InferenceServer {
public:
    InferenceServer(ModelManager& model_manager, const std::string& host = "localhost", int port = 8080);
    ~InferenceServer();

    // Start the server
    bool start();

    // Stop the server
    bool stop();

    // Generate text from a model
    // For non-streaming, use completion_callback
    // For streaming, use stream_callback
    std::string generate(const std::string& model_id, const std::string& prompt,
                        const InferenceConfig& config = InferenceConfig{},
                        std::function<void(const std::string&, bool)> stream_callback = nullptr,
                        std::function<void(const GenerationResult&)> completion_callback = nullptr);

    // Load a model
    bool loadModel(const std::string& model_id);

    // Unload a model
    bool unloadModel(const std::string& model_id);

    // Check if a model is loaded
    bool isModelLoaded(const std::string& model_id) const;

    // Get list of loaded models
    std::vector<std::string> getLoadedModels() const;

    // Get server status
    ServerStatus getStatus() const;

private:
    ModelManager& model_manager_;
    std::string host_;
    int port_;
    std::unique_ptr<httplib::Server> http_server_;
    std::atomic<bool> running_{false};

    // Thread for processing requests
    std::unique_ptr<std::thread> worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<InferenceRequest> request_queue_;
    std::atomic<bool> stop_requested_{false};

    // Model cache
    struct ModelInstance {
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;
        std::chrono::steady_clock::time_point last_used;
        std::atomic<bool> in_use{false};
        std::mutex mutex;
    };

    mutable std::mutex models_mutex_;
    std::unordered_map<std::string, std::unique_ptr<ModelInstance>> loaded_models_;

    // Server status
    mutable std::mutex status_mutex_;
    ServerStatus status_;

    // Maximum number of models to keep in memory
    size_t max_loaded_models_ = 3;

    // Worker thread function
    void workerThreadFunc();

    // Process a single inference request
    GenerationResult processRequest(const InferenceRequest& request);

    // Setup HTTP routes
    void setupHttpRoutes();

    // Initialize a loaded model with LoRA weights if available
    bool initializeLoraWeights(ModelInstance* instance, const std::string& model_id);

    // Load and cache a model for inference
    ModelInstance* getOrLoadModel(const std::string& model_id);

    // Unload least recently used model if we exceed max_loaded_models
    void unloadLeastRecentlyUsedModel();

    // Update server status
    void updateStatus(const std::function<void(ServerStatus&)>& updater);
};

} // namespace localllm

#endif // LOCALLLM_INFERENCE_SERVER_H
