#include "utils.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <regex>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <sysinfoapi.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// If libcurl is available
#if defined(HAVE_LIBCURL)
#include <curl/curl.h>
#endif

namespace localllm {
namespace utils {

// String utility functions
std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }

    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

std::string toLower(const std::string& s) {
    std::string result(s);
    std::transform(result.begin(), result.end(), result.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

// Random ID generation
std::string generateUniqueId(int length) {
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);

    std::string id;
    id.reserve(length);

    // Add timestamp component
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();
    id += std::to_string(value);

    // Add random component
    for (int i = id.length(); i < length; ++i) {
        id += chars[dis(gen)];
    }

    // Truncate if too long
    if (id.length() > static_cast<size_t>(length)) {
        id = id.substr(0, length);
    }

    return id;
}

// Time utilities
std::string getCurrentTimestamp(bool filename_safe) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;

    if (filename_safe) {
        ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
    } else {
        ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    }

    return ss.str();
}

std::string formatDuration(const std::chrono::milliseconds& duration) {
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    auto hours = total_seconds / 3600;
    auto minutes = (total_seconds % 3600) / 60;
    auto seconds = total_seconds % 60;

    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        ss << minutes << "m ";
    }
    ss << seconds << "s";

    return ss.str();
}

// File utilities
bool fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::optional<std::string> readTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool writeTextFile(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file) {
        return false;
    }

    file << content;
    return file.good();
}

std::optional<std::vector<uint8_t>> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    std::vector<uint8_t> buffer(file.tellg());
    file.seekg(0, std::ios::beg);

    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    if (!file.good()) {
        return std::nullopt;
    }

    return buffer;
}

bool writeBinaryFile(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return file.good();
}

// Network utilities
#if defined(HAVE_LIBCURL)
// Callback function for CURL to write data to a string
static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* response = static_cast<std::string*>(userp);
    size_t real_size = size * nmemb;
    response->append(static_cast<char*>(contents), real_size);
    return real_size;
}

// Callback function for CURL to write data to a file
static size_t writeFileCallback(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Callback function for CURL to report progress
static int progressCallback(void* clientp, double dltotal, double dlnow, double ultotal, double ulnow) {
    if (clientp && dltotal > 0) {
        auto* callback = reinterpret_cast<std::function<void(size_t, size_t)>*>(clientp);
        (*callback)(static_cast<size_t>(dlnow), static_cast<size_t>(dltotal));
    }
    return 0;  // Return non-zero to abort the transfer
}

bool downloadFile(const std::string& url, const std::string& destination_path,
                 std::function<void(size_t, size_t)> progress_callback) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    FILE* fp = fopen(destination_path.c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeFileCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    if (progress_callback) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progressCallback);
        curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &progress_callback);
    }

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);

    return res == CURLE_OK;
}
#else
bool downloadFile(const std::string& url, const std::string& destination_path,
                 std::function<void(size_t, size_t)> progress_callback) {
    // Libcurl not available, implement a simple fallback or return false
    return false;
}
#endif

bool isValidUrl(const std::string& url) {
    std::regex url_regex(
        R"(^(https?|ftp)://)"  // Protocol
        R"(([a-zA-Z0-9_-]+\.)+)"  // Subdomain
        R"([a-zA-Z0-9_-]+)"  // Domain
        R"((/[a-zA-Z0-9_\-\.~:/?#[\]@!$&'()*+,;=]*)?$)"  // Path, query, etc.
    );
    return std::regex_match(url, url_regex);
}

// Memory utilities
std::string formatMemorySize(size_t bytes) {
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && suffix_idx < 4) {
        size /= 1024;
        suffix_idx++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffix_idx];
    return ss.str();
}

size_t getAvailableMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullAvailPhys;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
#endif
}

size_t getTotalMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
#endif
}

// Hash utilities
// Simple implementation - in a real application, use a proper cryptographic library
std::string sha256(const std::string& input) {
    // Placeholder implementation - not cryptographically secure
    std::hash<std::string> hasher;
    auto hash = hasher(input);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

std::string md5(const std::string& input) {
    // Placeholder implementation - not cryptographically secure
    std::hash<std::string> hasher;
    auto hash = hasher(input);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

} // namespace utils
} // namespace localllm
