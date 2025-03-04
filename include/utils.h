#ifndef LOCALLLM_UTILS_H
#define LOCALLLM_UTILS_H

#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <random>

namespace localllm {
namespace utils {

// String utility functions
std::string trim(const std::string& s);
std::string toLower(const std::string& s);
std::vector<std::string> split(const std::string& s, char delimiter);
bool startsWith(const std::string& s, const std::string& prefix);
bool endsWith(const std::string& s, const std::string& suffix);
std::string replaceAll(std::string str, const std::string& from, const std::string& to);

// Random ID generation
std::string generateUniqueId(int length = 12);

// Time utilities
std::string getCurrentTimestamp(bool filename_safe = false);
std::string formatDuration(const std::chrono::milliseconds& duration);

// File utilities
bool fileExists(const std::string& path);
std::optional<std::string> readTextFile(const std::string& path);
bool writeTextFile(const std::string& path, const std::string& content);
std::optional<std::vector<uint8_t>> readBinaryFile(const std::string& path);
bool writeBinaryFile(const std::string& path, const std::vector<uint8_t>& data);

// Network utilities
bool downloadFile(const std::string& url, const std::string& destination_path,
                 std::function<void(size_t, size_t)> progress_callback = nullptr);
bool isValidUrl(const std::string& url);

// Memory utilities
std::string formatMemorySize(size_t bytes);
size_t getAvailableMemory();
size_t getTotalMemory();

// Hash utilities
std::string sha256(const std::string& input);
std::string md5(const std::string& input);

} // namespace utils
} // namespace localllm

#endif // LOCALLLM_UTILS_H
