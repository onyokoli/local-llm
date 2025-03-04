#include <iostream>
#include <stdexcept>
#include <string>
#include <filesystem>

#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"

#include "app.h"
#include "cli.h"
#include "version.h"

int main(int argc, char* argv[]) {
    try {
        // Set up logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        spdlog::set_level(spdlog::level::info);

        // Parse command line arguments
        std::filesystem::path base_dir = std::filesystem::current_path() / "localllm_data";
        uint16_t server_port = 8080;
        bool show_version = false;
        bool verbose = false;

        // Create argument parser using CLI11
        CLI::App app{"LocalLLM - Local Fine-tuning Infrastructure"};

        app.add_option("--dir", base_dir, "Base directory for data storage");
        app.add_option("--port", server_port, "Port for inference server");
        app.add_flag("--version", show_version, "Show version information and exit");
        app.add_flag("--verbose", verbose, "Enable verbose logging");

        // Parse arguments
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            return app.exit(e);
        }

        // Set verbose logging if requested
        if (verbose) {
            spdlog::set_level(spdlog::level::debug);
            spdlog::debug("Verbose logging enabled");
        }

        // Show version and exit if requested
        if (show_version) {
            std::cout << "LocalLLM version " << localllm::VERSION << std::endl;
            std::cout << "Build: " << localllm::BUILD_TYPE << " (" << localllm::BUILD_DATE << " " << localllm::BUILD_TIME << ")" << std::endl;
            std::cout << "Features: ";
            std::cout << "CUDA=" << (localllm::CUDA_SUPPORTED ? "yes" : "no") << ", ";
            std::cout << "OpenMP=" << (localllm::OPENMP_SUPPORTED ? "yes" : "no") << ", ";
            std::cout << "BLAS=" << (localllm::BLAS_SUPPORTED ? "yes" : "no") << std::endl;
            return 0;
        }

        // Create LocalLLM app
        spdlog::info("Initializing LocalLLM");
        spdlog::info("Base directory: {}", base_dir.string());
        spdlog::info("Server port: {}", server_port);

        localllm::LocalLLMApp llm_app(base_dir, server_port);

        // Create and start CLI
        localllm::CommandLineInterface cli(llm_app);
        cli.start();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
