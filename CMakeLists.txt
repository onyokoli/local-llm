cmake_minimum_required(VERSION 3.14)
project(LocalLLM VERSION 1.0.0 LANGUAGES CXX)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# On macOS, force CUDA OFF and use Metal instead
set(LOCALLLM_USE_CUDA OFF)
set(LOCALLLM_USE_METAL ON)

# Other options
option(LOCALLLM_BUILD_TESTS "Build tests" OFF)
option(LOCALLLM_USE_OPENMP "Enable OpenMP support for multi-threading" ON)
option(LOCALLLM_USE_BLAS "Enable BLAS support for linear algebra" ON)
option(LOCALLLM_USE_SYSTEM_DEPS "Use system-installed dependencies instead of fetching them" OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(include)

# Dependencies
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Force disable CUDA and enable Metal for llama.cpp
set(LLAMA_CUDA OFF CACHE BOOL "CUDA is disabled" FORCE)
set(LLAMA_METAL ON CACHE BOOL "Enable Metal for Apple Silicon" FORCE)

# Other llama.cpp options
set(LLAMA_NATIVE ON CACHE BOOL "Enable CPU architecture based optimizations" FORCE)
set(LLAMA_STATIC ON CACHE BOOL "Build llama.cpp as a static library" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Don't build llama.cpp examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Don't build llama.cpp tests" FORCE)
set(LLAMA_OPENMP ${LOCALLLM_USE_OPENMP} CACHE BOOL "OpenMP support" FORCE)
set(LLAMA_BLAS ${LOCALLLM_USE_BLAS} CACHE BOOL "BLAS support" FORCE)

# Fetch llama.cpp
FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(llama)

# nlohmann_json - JSON parsing
if(NOT LOCALLLM_USE_SYSTEM_DEPS)
    FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2
  )
    FetchContent_MakeAvailable(json)
endif()

# CLI11 - Command line argument parsing
if(NOT LOCALLLM_USE_SYSTEM_DEPS)
    FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.3.1
  )
    FetchContent_MakeAvailable(CLI11)
endif()

# spdlog - Logging library
if(NOT LOCALLLM_USE_SYSTEM_DEPS)
    FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.11.0
  )
    set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "Build spdlog as a shared library" FORCE)
    FetchContent_MakeAvailable(spdlog)
endif()

# cpp-httplib - HTTP server for REST API
if(NOT LOCALLLM_USE_SYSTEM_DEPS)
    FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.12.1
  )
    set(HTTPLIB_COMPILE ON CACHE BOOL "Build httplib as a library" FORCE)
    FetchContent_MakeAvailable(httplib)
endif()

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Define version information
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Source files - Add app.cpp
set(LOCALLLM_SOURCES
    src/main.cpp
    src/app.cpp
    src/model_manager.cpp
    src/dataset_manager.cpp
    src/fine_tuning_engine.cpp
    src/inference_server.cpp
    src/cli.cpp
    src/utils.cpp
)

# Header files
set(LOCALLLM_HEADERS
    include/app.h
    include/model_manager.h
    include/dataset_manager.h
    include/fine_tuning_engine.h
    include/inference_server.h
    include/cli.h
    include/utils.h
    include/types.h
)

# Main executable
add_executable(localllm ${LOCALLLM_SOURCES} ${LOCALLLM_HEADERS})

# Include paths for llama.cpp - CRITICAL
target_include_directories(localllm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${llama_SOURCE_DIR}         # Main llama.cpp directory
    ${llama_SOURCE_DIR}/common  # Common headers
)

# Dependencies
target_link_libraries(localllm
    PRIVATE
    llama
    nlohmann_json::nlohmann_json
    CLI11::CLI11
    spdlog::spdlog
    httplib::httplib
    SQLite::SQLite3
)

# Compile definitions
target_compile_definitions(localllm PRIVATE
    LOCALLLM_VERSION="${PROJECT_VERSION}"
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(localllm PRIVATE -Wall -Wextra -Werror=return-type)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(localllm PRIVATE /W4)
endif()

# Install targets
install(TARGETS localllm
    RUNTIME DESTINATION bin
)

# Package configuration
set(CPACK_PACKAGE_NAME "localllm")
set(CPACK_PACKAGE_VENDOR "LocalLLM Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Local Fine-tuning Infrastructure for LLMs")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "LocalLLM")

# OS-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "LocalLLM Team")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libsqlite3-0, libomp5")
endif()

include(CPack)

# Tests
if(LOCALLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA support: OFF (disabled for macOS)")
message(STATUS "Metal support: ON (enabled for Apple Silicon)")
message(STATUS "OpenMP support: ${LOCALLLM_USE_OPENMP}")
message(STATUS "BLAS support: ${LOCALLLM_USE_BLAS}")
message(STATUS "System dependencies: ${LOCALLLM_USE_SYSTEM_DEPS}")

# Print llama.cpp include paths to help with debugging
message(STATUS "llama.cpp include path: ${llama_SOURCE_DIR}")
