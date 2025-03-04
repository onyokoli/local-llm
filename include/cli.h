#ifndef LOCALLLM_CLI_H
#define LOCALLLM_CLI_H

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace localllm {

class LocalLLMApp;

class CommandLineInterface {
public:
    CommandLineInterface(LocalLLMApp& app);

    // Start the CLI
    void start();

private:
    LocalLLMApp& app_;
    bool running_ = true;

    // Command handlers
    using CommandHandler = std::function<void(const std::vector<std::string>&)>;
    std::unordered_map<std::string, CommandHandler> command_handlers_;

    // Process a command
    void processCommand(const std::string& input);

    // Command implementations
    void displayHelp();
    void listModels();
    void downloadModel(const std::vector<std::string>& args);
    void removeModel(const std::vector<std::string>& args);
    void listDatasets();
    void importDataset(const std::vector<std::string>& args);
    void startFineTuning(const std::vector<std::string>& args);
    void stopFineTuning();
    void generateText(const std::vector<std::string>& args);
    void showVersion();
};

} // namespace localllm

#endif // LOCALLLM_CLI_H
