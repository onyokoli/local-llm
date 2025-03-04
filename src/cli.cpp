#include "cli.h"
#include "app.h"
#include "version.h"
#include "utils.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace localllm {

CommandLineInterface::CommandLineInterface(LocalLLMApp& app) : app_(app) {
    // Register command handlers
    command_handlers_["help"] = [this](const std::vector<std::string>&) { displayHelp(); };
    command_handlers_["exit"] = [this](const std::vector<std::string>&) { running_ = false; };
    command_handlers_["quit"] = [this](const std::vector<std::string>&) { running_ = false; };

    command_handlers_["list-models"] = [this](const std::vector<std::string>&) { listModels(); };
    command_handlers_["download-model"] = [this](const std::vector<std::string>& args) { downloadModel(args); };
    command_handlers_["remove-model"] = [this](const std::vector<std::string>& args) { removeModel(args); };

    command_handlers_["list-datasets"] = [this](const std::vector<std::string>&) { listDatasets(); };
    command_handlers_["import-dataset"] = [this](const std::vector<std::string>& args) { importDataset(args); };

    command_handlers_["finetune"] = [this](const std::vector<std::string>& args) { startFineTuning(args); };
    command_handlers_["stop-finetune"] = [this](const std::vector<std::string>&) { stopFineTuning(); };

    command_handlers_["generate"] = [this](const std::vector<std::string>& args) { generateText(args); };
    command_handlers_["version"] = [this](const std::vector<std::string>&) { showVersion(); };
}

void CommandLineInterface::start() {
    app_.initialize();

    std::cout << "\nLocalLLM - Local Fine-tuning Infrastructure\n";
    std::cout << app_.getVersionInfo() << "\n";
    std::cout << "Enter 'help' for a list of commands or 'exit' to quit\n\n";

    while (running_) {
        std::string input;
        std::cout << "> ";
        std::getline(std::cin, input);

        if (input.empty()) continue;

        processCommand(input);
    }

    app_.shutdown();
}

void CommandLineInterface::processCommand(const std::string& input) {
    std::vector<std::string> parts;
    std::istringstream iss(input);
    std::string part;

    // Parse input respecting quoted strings
    while (iss >> std::quoted(part)) {
        parts.push_back(part);
    }

    if (parts.empty()) return;

    std::string command = parts[0];
    std::vector<std::string> args(parts.begin() + 1, parts.end());

    auto it = command_handlers_.find(command);
    if (it != command_handlers_.end()) {
        try {
            it->second(args);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Unknown command: " << command << std::endl;
        std::cout << "Type 'help' for a list of available commands" << std::endl;
    }
}

void CommandLineInterface::displayHelp() {
    std::cout << "Available commands:\n";
    std::cout << "  help                         - Display this help message\n";
    std::cout << "  version                      - Show version information\n";
    std::cout << "  exit, quit                   - Exit the application\n";
    std::cout << "\n";
    std::cout << "  list-models                  - List all available models\n";
    std::cout << "  download-model <id> [source] - Download a model (source: huggingface)\n";
    std::cout << "  download-model <id> --quantize [q4_0|q5_0] - Download and quantize\n";
    std::cout << "  remove-model <id> [--delete] - Remove a model (--delete to remove files)\n";
    std::cout << "\n";
    std::cout << "  list-datasets                - List all available datasets\n";
    std::cout << "  import-dataset <path> [name] [format] - Import a dataset\n";
    std::cout << "\n";
    std::cout << "  finetune <model-id> <dataset-id> [options] - Fine-tune a model\n";
    std::cout << "    Options:\n";
    std::cout << "      --method [full|lora|qlora] - Fine-tuning method (default: lora)\n";
    std::cout << "      --lr <value>              - Learning rate (default: 2e-5)\n";
    std::cout << "      --epochs <value>          - Number of epochs (default: 3)\n";
    std::cout << "      --batch-size <value>      - Batch size (default: 1)\n";
    std::cout << "      --lora-rank <value>       - LoRA rank (default: 8)\n";
    std::cout << "  stop-finetune                - Stop the current fine-tuning process\n";
    std::cout << "\n";
    std::cout << "  generate <model-id> \"<prompt>\" [options] - Generate text\n";
    std::cout << "    Options:\n";
    std::cout << "      --temp <value>           - Temperature (default: 0.8)\n";
    std::cout << "      --max-tokens <value>     - Maximum tokens to generate (default: 512)\n";
    std::cout << "      --top-p <value>          - Top-p sampling (default: 0.95)\n";
    std::cout << "      --top-k <value>          - Top-k sampling (default: 40)\n";
    std::cout << "\n";
}

void CommandLineInterface::listModels() {
    auto models = app_.listModels();

    if (models.empty()) {
        std::cout << "No models available. Use 'download-model' to download a model." << std::endl;
        return;
    }

    std::cout << "Available models:\n";
    for (const auto& model : models) {
        std::cout << "  - " << model.name << " (" << model.id << ")\n";
        std::cout << "    Architecture: " << model.architecture_type
                  << ", Parameters: " << (model.parameter_count / 1000000) << "M\n";

        if (model.is_quantized) {
            std::cout << "    Quantized: " << model.quantization_method << "\n";
        }

        if (model.is_finetuned) {
            std::cout << "    Fine-tuned: " << model.finetuning_method;
            if (model.finetuning_method == "lora" || model.finetuning_method == "qlora") {
                std::cout << " (rank " << model.lora_rank << ")";
            }
            std::cout << "\n";
            std::cout << "    Training dataset: " << model.training_dataset << "\n";
            std::cout << "    Training date: " << model.training_date << "\n";
        }

        std::cout << std::endl;
    }
}

void CommandLineInterface::downloadModel(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Error: Model ID required" << std::endl;
        std::cout << "Usage: download-model <id> [source] [--quantize [type]]" << std::endl;
        return;
    }

    std::string model_id = args[0];
    std::string source = "huggingface";
    bool quantize = false;
    std::string quantization_type = "q4_0";

    // Parse additional arguments
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--quantize") {
            quantize = true;
            if (i + 1 < args.size() && args[i + 1][0] != '-') {
                quantization_type = args[++i];
            }
        } else if (args[i][0] != '-') {
            source = args[i];
        }
    }

    std::cout << "Downloading model " << model_id << " from " << source;
    if (quantize) {
        std::cout << " (with " << quantization_type << " quantization)";
    }
    std::cout << "..." << std::endl;

    bool success = app_.downloadModel(model_id, source, quantize, quantization_type);

    if (success) {
        std::cout << "Model downloaded successfully" << std::endl;
    } else {
        std::cout << "Failed to download model" << std::endl;
    }
}

void CommandLineInterface::removeModel(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Error: Model ID required" << std::endl;
        std::cout << "Usage: remove-model <id> [--delete]" << std::endl;
        return;
    }

    std::string model_id = args[0];
    bool delete_files = false;

    // Check for --delete flag
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--delete") {
            delete_files = true;
        }
    }

    std::cout << "Removing model " << model_id;
    if (delete_files) {
        std::cout << " (including files)";
    }
    std::cout << "..." << std::endl;

    bool success = app_.removeModel(model_id, delete_files);

    if (success) {
        std::cout << "Model removed successfully" << std::endl;
    } else {
        std::cout << "Failed to remove model" << std::endl;
    }
}

void CommandLineInterface::listDatasets() {
    auto datasets = app_.listDatasets();

    if (datasets.empty()) {
        std::cout << "No datasets available. Use 'import-dataset' to import a dataset." << std::endl;
        return;
    }

    std::cout << "Available datasets:\n";
    for (const auto& dataset : datasets) {
        std::cout << "  - " << dataset.name << " (" << dataset.id << ")\n";
        std::cout << "    Format: " << dataset.format_type << "\n";
        std::cout << "    Examples: " << dataset.num_examples;

        if (dataset.is_processed) {
            std::cout << " (" << dataset.train_examples << " train, "
                     << dataset.val_examples << " validation)";
        }

        std::cout << "\n";
        std::cout << "    Tokens: " << dataset.total_tokens << "\n";
        std::cout << "    Created: " << dataset.created_date << "\n";
        std::cout << "    Processed: " << (dataset.is_processed ? "Yes" : "No") << "\n";
        std::cout << std::endl;
    }
}

void CommandLineInterface::importDataset(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Error: Dataset path required" << std::endl;
        std::cout << "Usage: import-dataset <path> [name] [format]" << std::endl;
        return;
    }

    std::string path = args[0];
    std::string name = "";
    std::string format = "jsonl";

    if (args.size() > 1) {
        name = args[1];
    }

    if (args.size() > 2) {
        format = args[2];
    }

    std::cout << "Importing dataset from " << path << "..." << std::endl;

    std::string dataset_id = app_.importDataset(path, name, format);

    if (!dataset_id.empty()) {
        std::cout << "Dataset imported successfully (ID: " << dataset_id << ")" << std::endl;
    } else {
        std::cout << "Failed to import dataset" << std::endl;
    }
}

void CommandLineInterface::startFineTuning(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "Error: Model ID and dataset ID required" << std::endl;
        std::cout << "Usage: finetune <model-id> <dataset-id> [options]" << std::endl;
        return;
    }

    std::string model_id = args[0];
    std::string dataset_id = args[1];

    // Create default config
    FineTuneConfig config;

    // Parse additional arguments
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--method") {
            if (i + 1 < args.size()) {
                std::string method = args[++i];
                if (method == "full") {
                    config.method = FineTuneConfig::Method::FULL_FINETUNE;
                } else if (method == "lora") {
                    config.method = FineTuneConfig::Method::LORA;
                } else if (method == "qlora") {
                    config.method = FineTuneConfig::Method::QLORA;
                } else {
                    std::cout << "Warning: Unknown method '" << method << "', using default" << std::endl;
                }
            }
        } else if (args[i] == "--lr") {
            if (i + 1 < args.size()) {
                try {
                    config.learning_rate = std::stof(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid learning rate, using default" << std::endl;
                }
            }
        } else if (args[i] == "--epochs") {
            if (i + 1 < args.size()) {
                try {
                    config.epochs = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid epochs value, using default" << std::endl;
                }
            }
        } else if (args[i] == "--batch-size") {
            if (i + 1 < args.size()) {
                try {
                    config.batch_size = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid batch size, using default" << std::endl;
                }
            }
        } else if (args[i] == "--lora-rank") {
            if (i + 1 < args.size()) {
                try {
                    config.lora_rank = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid LoRA rank, using default" << std::endl;
                }
            }
        }
    }

    std::cout << "Starting fine-tuning of model " << model_id << " with dataset " << dataset_id << "..." << std::endl;

    bool success = app_.startFineTuning(model_id, dataset_id, config);

    if (success) {
        std::cout << "Fine-tuning started successfully" << std::endl;
        std::cout << "Use 'stop-finetune' to stop the process" << std::endl;
    } else {
        std::cout << "Failed to start fine-tuning" << std::endl;
    }
}

void CommandLineInterface::stopFineTuning() {
    std::cout << "Stopping fine-tuning process..." << std::endl;
    app_.stopFineTuning();
    std::cout << "Stop request sent. The process will finish the current step and then stop." << std::endl;
}

void CommandLineInterface::generateText(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "Error: Model ID and prompt required" << std::endl;
        std::cout << "Usage: generate <model-id> \"<prompt>\" [options]" << std::endl;
        return;
    }

    std::string model_id = args[0];
    std::string prompt = args[1];

    // Create default config
    InferenceConfig config;

    // Parse additional arguments
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--temp") {
            if (i + 1 < args.size()) {
                try {
                    config.temperature = std::stof(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid temperature, using default" << std::endl;
                }
            }
        } else if (args[i] == "--max-tokens") {
            if (i + 1 < args.size()) {
                try {
                    config.max_tokens = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid max tokens value, using default" << std::endl;
                }
            }
        } else if (args[i] == "--top-p") {
            if (i + 1 < args.size()) {
                try {
                    config.top_p = std::stof(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid top-p value, using default" << std::endl;
                }
            }
        } else if (args[i] == "--top-k") {
            if (i + 1 < args.size()) {
                try {
                    config.top_k = std::stoi(args[++i]);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid top-k value, using default" << std::endl;
                }
            }
        }
    }

    std::cout << "Generating text with model " << model_id << "...\n" << std::endl;

    // For CLI we would implement a direct generation method
    // In a real implementation, we would call the inference server API
    std::cout << "NOTE: This is a placeholder. In a real implementation, text would be generated here.\n" << std::endl;
    std::cout << "Generated text would appear here..." << std::endl;
}

void CommandLineInterface::showVersion() {
    std::cout << app_.getVersionInfo() << std::endl;
    std::cout << "Build: " << localllm::BUILD_TYPE << " (" << localllm::BUILD_DATE << " " << localllm::BUILD_TIME << ")" << std::endl;
    std::cout << "Features: ";
    std::cout << "CUDA=" << (localllm::CUDA_SUPPORTED ? "yes" : "no") << ", ";
    std::cout << "OpenMP=" << (localllm::OPENMP_SUPPORTED ? "yes" : "no") << ", ";
    std::cout << "BLAS=" << (localllm::BLAS_SUPPORTED ? "yes" : "no") << std::endl;
}

} // namespace localllm
