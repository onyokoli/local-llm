#include "inference_server.h"
#include "model_manager.h"
#include "utils.h"

#include <chrono>
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thread>
#include <memory>
#include <functional>
#include <future>

#include "llama.h"
#include "ggml.h"
#include "httplib.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace localllm {

// Helper function to parse JSON request from httplib
bool parseJsonRequest(const httplib::Request& req, json& json_data, httplib::Response& res) {
    try {
        json_data = json::parse(req.body);
        return true;
    } catch (const json::exception& e) {
        res.status = 400;
        json error = {{"error", "Invalid JSON format"}, {"details", e.what()}};
        res.set_content(error.dump(), "application/json");
        return false;
    }
}

// Constructor
InferenceServer::InferenceServer(ModelManager& model_manager, const std::string& host, int port)
    : model_manager_(model_manager), host_(host), port_(port) {

    status_.host = host;
    status_.port = port;
    status_.running = false;
    status_.start_time = std::chrono::steady_clock::now();
}

// Destructor
InferenceServer::~InferenceServer() {
    stop();
}

// Start the server
bool InferenceServer::start() {
    if (running_) {
        spdlog::error("Server is already running");
        return false;
    }

    // Initialize llama backend
    llama_backend_init(false);

    // Create HTTP server
    http_server_ = std::make_unique<httplib::Server>();

    // Setup HTTP routes
    setupHttpRoutes();

    // Start worker thread
    stop_requested_ = false;
    worker_thread_ = std::make_unique<std::thread>(&InferenceServer::workerThreadFunc, this);

    // Start HTTP server in a new thread
    std::thread server_thread([this]() {
        spdlog::info("Starting HTTP server on {}:{}", host_, port_);
        if (!http_server_->listen(host_.c_str(), port_)) {
            spdlog::error("Failed to start HTTP server");
        }
    });

    // Detach server thread to run independently
    server_thread.detach();

    running_ = true;

    updateStatus([](ServerStatus& status) {
        status.running = true;
        status.start_time = std::chrono::steady_clock::now();
    });

    spdlog::info("Inference server started");
    return true;
}

// Stop the server
bool InferenceServer::stop() {
    if (!running_) {
        spdlog::info("Server is not running");
        return false;
    }

    spdlog::info("Stopping inference server");

    // Stop HTTP server
    if (http_server_) {
        http_server_->stop();
    }

    // Stop worker thread
    stop_requested_ = true;
    queue_cv_.notify_all();

    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }

    // Unload all models
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        for (auto& [id, instance] : loaded_models_) {
            std::lock_guard<std::mutex> model_lock(instance->mutex);
            if (instance->ctx) {
                llama_free(instance->ctx);
                instance->ctx = nullptr;
            }
            if (instance->model) {
                llama_free_model(instance->model);
                instance->model = nullptr;
            }
        }
        loaded_models_.clear();
    }

    // Update status
    updateStatus([](ServerStatus& status) {
        status.running = false;
        status.loaded_models.clear();
    });

    running_ = false;
    spdlog::info("Inference server stopped");

    return true;
}

// Generate text from a model
std::string InferenceServer::generate(const std::string& model_id, const std::string& prompt,
                                    const InferenceConfig& config,
                                    std::function<void(const std::string&, bool)> stream_callback,
                                    std::function<void(const GenerationResult&)> completion_callback) {
    // Create request ID
    std::string request_id = utils::generateUniqueId();

    spdlog::info("Generation request {}: model={}, prompt_length={}",
                 request_id, model_id, prompt.length());

    // Create inference request
    InferenceRequest request;
    request.id = request_id;
    request.model_id = model_id;
    request.prompt = prompt;
    request.config = config;
    request.created_at = std::chrono::steady_clock::now();
    request.stream_callback = stream_callback;
    request.completion_callback = completion_callback;

    // Update status
    updateStatus([](ServerStatus& status) {
        status.pending_requests++;
    });

    // Add to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(request);
    }

    // Notify worker thread
    queue_cv_.notify_one();

    return request_id;
}

// Load a model
bool InferenceServer::loadModel(const std::string& model_id) {
    spdlog::info("Loading model: {}", model_id);

    // Check if model exists
    if (!model_manager_.isModelAvailable(model_id)) {
        spdlog::error("Model not found: {}", model_id);
        return false;
    }

    // Check if model is already loaded
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        if (loaded_models_.find(model_id) != loaded_models_.end()) {
            spdlog::info("Model already loaded: {}", model_id);
            return true;
        }
    }

    // Get or load model
    try {
        ModelInstance* instance = getOrLoadModel(model_id);
        if (!instance) {
            spdlog::error("Failed to load model: {}", model_id);
            return false;
        }

        spdlog::info("Model loaded successfully: {}", model_id);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error loading model {}: {}", model_id, e.what());
        return false;
    }
}

// Unload a model
bool InferenceServer::unloadModel(const std::string& model_id) {
    spdlog::info("Unloading model: {}", model_id);

    // Check if model is loaded
    std::lock_guard<std::mutex> lock(models_mutex_);
    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        spdlog::info("Model not loaded: {}", model_id);
        return false;
    }

    // Check if model is in use
    if (it->second->in_use) {
        spdlog::error("Cannot unload model as it is currently in use: {}", model_id);
        return false;
    }

    // Unload model
    {
        std::lock_guard<std::mutex> model_lock(it->second->mutex);
        if (it->second->ctx) {
            llama_free(it->second->ctx);
            it->second->ctx = nullptr;
        }
        if (it->second->model) {
            llama_free_model(it->second->model);
            it->second->model = nullptr;
        }
    }

    // Remove from loaded models
    loaded_models_.erase(it);

    // Update status
    updateStatus([&model_id](ServerStatus& status) {
        status.loaded_models.erase(
            std::remove(status.loaded_models.begin(), status.loaded_models.end(), model_id),
            status.loaded_models.end()
        );
    });

    spdlog::info("Model unloaded successfully: {}", model_id);
    return true;
}

// Check if a model is loaded
bool InferenceServer::isModelLoaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    return loaded_models_.find(model_id) != loaded_models_.end();
}

// Get list of loaded models
std::vector<std::string> InferenceServer::getLoadedModels() const {
    std::lock_guard<std::mutex> lock(models_mutex_);
    std::vector<std::string> models;
    for (const auto& [id, _] : loaded_models_) {
        models.push_back(id);
    }
    return models;
}

// Get server status
ServerStatus InferenceServer::getStatus() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

// Worker thread function
void InferenceServer::workerThreadFunc() {
    spdlog::info("Worker thread started");

    while (!stop_requested_) {
        InferenceRequest request;
        bool has_request = false;

        // Get request from queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (request_queue_.empty()) {
                // Wait for new request or stop signal
                queue_cv_.wait_for(lock, std::chrono::seconds(1),
                                  [this]() { return !request_queue_.empty() || stop_requested_; });

                if (stop_requested_) break;
                if (request_queue_.empty()) continue;
            }

            request = request_queue_.front();
            request_queue_.pop();
            has_request = true;

            // Update status
            updateStatus([](ServerStatus& status) {
                status.pending_requests--;
                status.active_requests++;
            });
        }

        // Process request
        if (has_request) {
            spdlog::info("Processing request {}: model={}", request.id, request.model_id);

            // Process request and capture result
            auto start_time = std::chrono::steady_clock::now();
            GenerationResult result = processRequest(request);
            auto end_time = std::chrono::steady_clock::now();

            // Calculate generation time
            result.generation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

            // Call completion callback if provided
            if (request.completion_callback) {
                request.completion_callback(result);
            }

            // Log result
            if (result.success) {
                spdlog::info("Request {} completed: tokens={}, time={}ms",
                            request.id, result.token_count, result.generation_time_ms);
            } else {
                spdlog::error("Request {} failed: {}", request.id, result.error_message);
            }

            // Update status
            updateStatus([](ServerStatus& status) {
                status.active_requests--;
                status.completed_requests++;
            });
        }
    }

    spdlog::info("Worker thread stopped");
}

// Process a single inference request
GenerationResult InferenceServer::processRequest(const InferenceRequest& request) {
    GenerationResult result;
    result.success = false;
    result.text = "";
    result.token_count = 0;

    try {
        // Get or load model
        ModelInstance* instance = getOrLoadModel(request.model_id);
        if (!instance) {
            result.error_message = "Failed to load model: " + request.model_id;
            return result;
        }

        // Lock the model instance for exclusive use
        std::lock_guard<std::mutex> model_lock(instance->mutex);
        instance->in_use = true;
        instance->last_used = std::chrono::steady_clock::now();

        // Create a cleanup function to reset in_use flag
        auto cleanup = [instance]() {
            instance->in_use = false;
        };

        // Ensure cleanup happens even if an exception is thrown
        struct ScopeGuard {
            std::function<void()> func;
            ~ScopeGuard() { func(); }
        } guard{cleanup};

        // Reset the context for a new generation
        llama_reset_timings(instance->ctx);

        // Create the prompt
        const auto& config = request.config;

        // Tokenize the prompt
        std::vector<llama_token> tokens = llama_tokenize(instance->ctx, request.prompt.c_str(), true);

        if (tokens.empty()) {
            result.error_message = "Failed to tokenize prompt";
            return result;
        }

        // Check if prompt is too long
        if (static_cast<int>(tokens.size()) > config.context_length) {
            spdlog::warn("Prompt is too long ({} tokens), truncating to {} tokens",
                        tokens.size(), config.context_length);
            tokens.resize(config.context_length);
        }

        // Set up generation parameters
        llama_sampling_params sampling_params;
        sampling_params.n_prev = tokens.size();
        sampling_params.n_ctx = config.context_length;
        sampling_params.temp = config.temperature;
        sampling_params.top_p = config.top_p;
        sampling_params.top_k = config.top_k;
        sampling_params.repeat_penalty = config.repetition_penalty;
        sampling_params.tfs_z = 1.0f;
        sampling_params.typical_p = 1.0f;

        // Create sampling context
        llama_sampling_context* sampling_ctx = llama_sampling_init(sampling_params);

        // Process prompt tokens
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (stop_requested_) {
                result.error_message = "Request interrupted";
                llama_sampling_free(sampling_ctx);
                return result;
            }

            // Process token in batch
            int n_batch = config.n_batch;
            size_t batch_size = std::min(n_batch, static_cast<int>(tokens.size() - i));

            llama_batch batch = llama_batch_get_one(&tokens[i], batch_size, i, 0);

            if (llama_decode(instance->ctx, batch) != 0) {
                result.error_message = "Failed to decode prompt";
                llama_sampling_free(sampling_ctx);
                return result;
            }

            i += batch_size - 1;
        }

        // Generate response
        std::string generated_text;
        std::vector<llama_token> generated_tokens;

        bool stop_generation = false;
        int n_gen = 0;

        while (n_gen < config.max_tokens && !stop_generation && !stop_requested_) {
            // Sample next token
            llama_token token = llama_sampling_sample(sampling_ctx, instance->ctx, nullptr);
            n_gen++;

            // Stop on EOS or if any stop tokens are generated
            if (token == llama_token_eos(instance->model)) {
                stop_generation = true;
                break;
            }

            // Save token
            generated_tokens.push_back(token);

            // Convert token to text
            const char* token_text = llama_token_to_piece(instance->ctx, token);
            if (token_text == nullptr) {
                result.error_message = "Failed to convert token to text";
                llama_sampling_free(sampling_ctx);
                return result;
            }

            std::string token_str(token_text);
            generated_text += token_str;

            // Handle streaming
            if (request.stream_callback) {
                request.stream_callback(token_str, false);
            }

            // Feed the token back for the next iteration
            llama_batch batch = llama_batch_get_one(&token, 1, tokens.size() + n_gen - 1, 0);

            if (llama_decode(instance->ctx, batch) != 0) {
                result.error_message = "Failed to decode generated token";
                llama_sampling_free(sampling_ctx);
                return result;
            }
        }

        // Send final streaming callback if needed
        if (request.stream_callback) {
            request.stream_callback("", true);
        }

        // Clean up
        llama_sampling_free(sampling_ctx);

        // Set result
        result.success = true;
        result.text = generated_text;
        result.token_count = n_gen;

        return result;

    } catch (const std::exception& e) {
        result.error_message = std::string("Error during generation: ") + e.what();
        return result;
    }
}

// Setup HTTP routes
void InferenceServer::setupHttpRoutes() {
    if (!http_server_) return;

    // API status
    http_server_->Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
        ServerStatus status = getStatus();

        json response = {
            {"running", status.running},
            {"active_requests", status.active_requests},
            {"pending_requests", status.pending_requests},
            {"completed_requests", status.completed_requests},
            {"loaded_models", status.loaded_models},
            {"uptime_seconds", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - status.start_time).count()},
            {"memory_usage", utils::formatMemorySize(status.memory_usage)}
        };

        res.set_content(response.dump(), "application/json");
    });

    // List models
    http_server_->Get("/api/models", [this](const httplib::Request&, httplib::Response& res) {
        // Get all available models
        std::vector<ModelInfo> models = model_manager_.listAvailableModels();

        // Get loaded models
        std::vector<std::string> loaded_models = getLoadedModels();
        std::unordered_set<std::string> loaded_set(loaded_models.begin(), loaded_models.end());

        // Create response
        json response = json::array();

        for (const auto& model : models) {
            response.push_back({
                {"id", model.id},
                {"name", model.name},
                {"architecture", model.architecture_type},
                {"parameter_count", model.parameter_count},
                {"is_quantized", model.is_quantized},
                {"quantization_method", model.quantization_method},
                {"is_finetuned", model.is_finetuned},
                {"is_loaded", loaded_set.count(model.id) > 0}
            });
        }

        res.set_content(response.dump(), "application/json");
    });

    // Get model info
    http_server_->Get("/api/models/:id", [this](const httplib::Request& req, httplib::Response& res) {
        std::string model_id = req.path_params.at("id");

        // Get model info
        auto model_info_opt = model_manager_.getModelInfo(model_id);
        if (!model_info_opt) {
            res.status = 404;
            json error = {{"error", "Model not found"}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        const ModelInfo& model = *model_info_opt;

        // Check if model is loaded
        bool is_loaded = isModelLoaded(model_id);

        // Create response
        json response = {
            {"id", model.id},
            {"name", model.name},
            {"architecture", model.architecture_type},
            {"parameter_count", model.parameter_count},
            {"is_quantized", model.is_quantized},
            {"quantization_method", model.quantization_method},
            {"is_finetuned", model.is_finetuned},
            {"is_loaded", is_loaded}
        };

        if (model.is_finetuned) {
            response["parent_model"] = model.parent_model_id.value_or("");
            response["fine_tuning_method"] = model.finetuning_method;
            response["fine_tuning_dataset"] = model.training_dataset;
            response["fine_tuning_date"] = model.training_date;
        }

        res.set_content(response.dump(), "application/json");
    });

    // Load model
    http_server_->Post("/api/models/:id/load", [this](const httplib::Request& req, httplib::Response& res) {
        std::string model_id = req.path_params.at("id");

        // Check if model exists
        if (!model_manager_.isModelAvailable(model_id)) {
            res.status = 404;
            json error = {{"error", "Model not found"}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Load model
        bool success = loadModel(model_id);

        if (success) {
            json response = {{"success", true}, {"model_id", model_id}};
            res.set_content(response.dump(), "application/json");
        } else {
            res.status = 500;
            json error = {{"error", "Failed to load model"}, {"model_id", model_id}};
            res.set_content(error.dump(), "application/json");
        }
    });

    // Unload model
    http_server_->Post("/api/models/:id/unload", [this](const httplib::Request& req, httplib::Response& res) {
        std::string model_id = req.path_params.at("id");

        // Check if model is loaded
        if (!isModelLoaded(model_id)) {
            res.status = 404;
            json error = {{"error", "Model not loaded"}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Unload model
        bool success = unloadModel(model_id);

        if (success) {
            json response = {{"success", true}, {"model_id", model_id}};
            res.set_content(response.dump(), "application/json");
        } else {
            res.status = 500;
            json error = {{"error", "Failed to unload model"}, {"model_id", model_id}};
            res.set_content(error.dump(), "application/json");
        }
    });

    // Generate text
    http_server_->Post("/api/models/:id/generate", [this](const httplib::Request& req, httplib::Response& res) {
        std::string model_id = req.path_params.at("id");

        // Parse request JSON
        json json_data;
        if (!parseJsonRequest(req, json_data, res)) {
            return;
        }

        // Validate required fields
        if (!json_data.contains("prompt") || !json_data["prompt"].is_string()) {
            res.status = 400;
            json error = {{"error", "Missing or invalid 'prompt' field"}};
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string prompt = json_data["prompt"];

        // Create inference config with defaults
        InferenceConfig config;

        // Override with request values if provided
        if (json_data.contains("temperature") && json_data["temperature"].is_number()) {
            config.temperature = json_data["temperature"];
        }

        if (json_data.contains("max_tokens") && json_data["max_tokens"].is_number_integer()) {
            config.max_tokens = json_data["max_tokens"];
        }

        if (json_data.contains("top_p") && json_data["top_p"].is_number()) {
            config.top_p = json_data["top_p"];
        }

        if (json_data.contains("top_k") && json_data["top_k"].is_number_integer()) {
            config.top_k = json_data["top_k"];
        }

        if (json_data.contains("repetition_penalty") && json_data["repetition_penalty"].is_number()) {
            config.repetition_penalty = json_data["repetition_penalty"];
        }

        if (json_data.contains("seed") && json_data["seed"].is_number_integer()) {
            config.seed = json_data["seed"];
        }

        // Check streaming flag
        bool is_streaming = false;
        if (json_data.contains("stream") && json_data["stream"].is_boolean()) {
            is_streaming = json_data["stream"];
            config.stream = is_streaming;
        }

        // For streaming response
        if (is_streaming) {
            // Set response headers for server-sent events
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            // Open connection for streaming
            auto stream_callback = [&res](const std::string& token, bool done) {
                json event;

                if (!done) {
                    event = {
                        {"type", "token"},
                        {"token", token}
                    };
                } else {
                    event = {
                        {"type", "done"}
                    };
                }

                std::string event_data = "data: " + event.dump() + "\n\n";
                res.write(event_data.c_str());
                res.flush();
            };

            // Start generation (doesn't wait for completion)
            std::string request_id = generate(model_id, prompt, config, stream_callback, nullptr);

        } else {
            // For non-streaming response, use promise/future to wait for result
            std::promise<GenerationResult> result_promise;
            std::future<GenerationResult> result_future = result_promise.get_future();

            auto completion_callback = [&result_promise](const GenerationResult& result) {
                result_promise.set_value(result);
            };

            // Start generation
            std::string request_id = generate(model_id, prompt, config, nullptr, completion_callback);

            // Wait for result
            GenerationResult result = result_future.get();

            // Send response
            if (result.success) {
                json response = {
                    {"id", request_id},
                    {"model", model_id},
                    {"text", result.text},
                    {"tokens", result.token_count},
                    {"generation_time_ms", result.generation_time_ms}
                };
                res.set_content(response.dump(), "application/json");
            } else {
                res.status = 500;
                json error = {
                    {"error", "Generation failed"},
                    {"message", result.error_message}
                };
                res.set_content(error.dump(), "application/json");
            }
        }
    });
}

// Initialize a loaded model with LoRA weights if available
bool InferenceServer::initializeLoraWeights(ModelInstance* instance, const std::string& model_id) {
    // Get model info
    auto model_info_opt = model_manager_.getModelInfo(model_id);
    if (!model_info_opt) {
        return false;
    }

    const ModelInfo& model_info = *model_info_opt;

    // Check if model is fine-tuned with LoRA
    if (!model_info.is_finetuned ||
        (model_info.finetuning_method != "lora" && model_info.finetuning_method != "qlora")) {
        return true;  // Not a LoRA model, nothing to do
    }

    spdlog::info("Initializing LoRA weights for model {}", model_id);

    // Find LoRA weights file
    fs::path model_path = model_manager_.getModelPath(model_id);
    fs::path lora_weights_path = model_path / "lora_weights.bin";

    if (!fs::exists(lora_weights_path)) {
        spdlog::error("LoRA weights file not found: {}", lora_weights_path.string());
        return false;
    }

    // Open weights file
    std::ifstream weights_file(lora_weights_path, std::ios::binary);
    if (!weights_file.is_open()) {
        spdlog::error("Failed to open LoRA weights file: {}", lora_weights_path.string());
        return false;
    }

    // Read header
    uint32_t num_tensors = 0;
    weights_file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    // Read tensors
    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Read tensor name
        uint32_t name_length = 0;
        weights_file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));

        std::string name(name_length, '\0');
        weights_file.read(&name[0], name_length);

        // Read tensor dimensions
        uint32_t n_dims = 0;
        weights_file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));

        std::vector<uint32_t> dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            weights_file.read(reinterpret_cast<char*>(&dims[d]), sizeof(uint32_t));
        }

        // Create tensor in the model
        ggml_type dtype = GGML_TYPE_F32;  // LoRA weights are typically F32

        ggml_context* ctx = llama_get_ggml_context(instance->ctx);
        if (!ctx) {
            spdlog::error("Failed to get GGML context");
            return false;
        }

        ggml_tensor* tensor = nullptr;

        if (n_dims == 1) {
            tensor = ggml_new_tensor_1d(ctx, dtype, dims[0]);
        } else if (n_dims == 2) {
            tensor = ggml_new_tensor_2d(ctx, dtype, dims[0], dims[1]);
        } else if (n_dims == 3) {
            tensor = ggml_new_tensor_3d(ctx, dtype, dims[0], dims[1], dims[2]);
        } else {
            spdlog::error("Unsupported tensor dimensionality: {}", n_dims);
            continue;
        }

        if (!tensor) {
            spdlog::error("Failed to create tensor for {}", name);
            continue;
        }

        // Read tensor data
        size_t data_size = ggml_nbytes(tensor);
        weights_file.read(reinterpret_cast<char*>(tensor->data), data_size);

        // Add tensor to model
        if (llama_model_tensors_add(instance->model, name.c_str(), tensor) != 0) {
            spdlog::error("Failed to add tensor {} to model", name);
            continue;
        }
    }

    weights_file.close();

    spdlog::info("LoRA weights initialized successfully");
    return true;
}

// Load and cache a model for inference
InferenceServer::ModelInstance* InferenceServer::getOrLoadModel(const std::string& model_id) {
    // Check if model is already loaded
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        auto it = loaded_models_.find(model_id);
        if (it != loaded_models_.end()) {
            return it->second.get();
        }
    }

    // Check if model exists
    if (!model_manager_.isModelAvailable(model_id)) {
        spdlog::error("Model not found: {}", model_id);
        return nullptr;
    }

    // Get model info
    auto model_info_opt = model_manager_.getModelInfo(model_id);
    if (!model_info_opt) {
        spdlog::error("Failed to get model info for {}", model_id);
        return nullptr;
    }

    // Find model file
    fs::path model_path = model_manager_.getModelPath(model_id);

    fs::path model_file;
    for (const auto& entry : fs::directory_iterator(model_path)) {
        if (entry.path().extension() == ".bin" ||
            entry.path().extension() == ".gguf" ||
            entry.path().extension() == ".ggml") {
            model_file = entry.path();
            break;
        }
    }

    if (model_file.empty()) {
        spdlog::error("Could not find model file in {}", model_path.string());
        return nullptr;
    }

    spdlog::info("Loading model from {}", model_file.string());

    // Unload least recently used model if we've reached the limit
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        if (loaded_models_.size() >= max_loaded_models_) {
            unloadLeastRecentlyUsedModel();
        }
    }

    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_file.c_str(), model_params);

    if (!model) {
        spdlog::error("Failed to load model from {}", model_file.string());
        return nullptr;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model_info_opt->context_length;
    ctx_params.n_batch = 512;  // Default batch size

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (!ctx) {
        spdlog::error("Failed to create context for model {}", model_id);
        llama_free_model(model);
        return nullptr;
    }

    // Create model instance
    auto instance = std::make_unique<ModelInstance>();
    instance->model = model;
    instance->ctx = ctx;
    instance->last_used = std::chrono::steady_clock::now();
    instance->in_use = false;

    // Initialize LoRA weights if applicable
    if (!initializeLoraWeights(instance.get(), model_id)) {
        spdlog::warn("Failed to initialize LoRA weights for model {}", model_id);
        // Continue without LoRA
    }

    // Add to loaded models
    ModelInstance* instance_ptr = instance.get();

    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        loaded_models_[model_id] = std::move(instance);

        // Update status
        updateStatus([&model_id](ServerStatus& status) {
            status.loaded_models.push_back(model_id);
        });
    }

    spdlog::info("Model {} loaded successfully", model_id);
    return instance_ptr;
}

// Unload least recently used model if we exceed max_loaded_models
void InferenceServer::unloadLeastRecentlyUsedModel() {
    // This function should be called with models_mutex_ already locked

    if (loaded_models_.empty()) {
        return;
    }

    // Find least recently used model that is not in use
    std::string lru_model_id;
    std::chrono::steady_clock::time_point oldest_time = std::chrono::steady_clock::now();

    for (const auto& [id, instance] : loaded_models_) {
        if (!instance->in_use && instance->last_used < oldest_time) {
            oldest_time = instance->last_used;
            lru_model_id = id;
        }
    }

    if (lru_model_id.empty()) {
        spdlog::warn("All loaded models are in use, cannot unload any");
        return;
    }

    // Unload the model
    auto it = loaded_models_.find(lru_model_id);
    if (it != loaded_models_.end()) {
        spdlog::info("Unloading least recently used model: {}", lru_model_id);

        if (it->second->ctx) {
            llama_free(it->second->ctx);
            it->second->ctx = nullptr;
        }

        if (it->second->model) {
            llama_free_model(it->second->model);
            it->second->model = nullptr;
        }

        loaded_models_.erase(it);

        // Update status
        updateStatus([&lru_model_id](ServerStatus& status) {
            status.loaded_models.erase(
                std::remove(status.loaded_models.begin(), status.loaded_models.end(), lru_model_id),
                status.loaded_models.end()
            );
        });
    }
}

// Update server status
void InferenceServer::updateStatus(const std::function<void(ServerStatus&)>& updater) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    updater(status_);

    // Update memory usage
    status_.memory_usage = 0;
    for (const auto& [_, instance] : loaded_models_) {
        if (instance->model) {
            status_.memory_usage += llama_model_size(instance->model);
        }

        if (instance->ctx) {
            status_.memory_usage += llama_get_state_size(instance->ctx);
        }
    }
}

} // namespace localllm
