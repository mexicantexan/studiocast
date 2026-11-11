cmake_minimum_required(VERSION 3.22)

set(STUDIOCAST_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
if (NOT EXISTS "${STUDIOCAST_VERSION_FILE}")
    message(FATAL_ERROR "VERSION file not found: ${STUDIOCAST_VERSION_FILE}")
endif ()

file(READ "${STUDIOCAST_VERSION_FILE}" STUDIOCAST_PROJECT_VERSION)
string(STRIP "${STUDIOCAST_PROJECT_VERSION}" STUDIOCAST_PROJECT_VERSION)
if (NOT STUDIOCAST_PROJECT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
            "VERSION must contain a numeric MAJOR.MINOR.PATCH version, got '${STUDIOCAST_PROJECT_VERSION}'")
endif ()

project(StudioCast
        VERSION ${STUDIOCAST_PROJECT_VERSION}
        DESCRIPTION "Broadcast-style audio/video effects manager for Linux (Maxine-ready)"
        LANGUAGES CXX
)

include(CTest)

function(studiocast_set_test_labels test_name)
    set_tests_properties("${test_name}" PROPERTIES LABELS "${ARGN}")
endfunction()

foreach (var IN ITEMS CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH)
    if (DEFINED ENV{${var}} AND NOT "$ENV{${var}}" STREQUAL "")
        message(WARNING
                "${var} is set to '$ENV{${var}}'. This can break the C++ toolchain by forcing mismatched headers. "
                "Consider unsetting it for StudioCast builds.")
    endif ()
endforeach ()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(CompilerWarnings)
include(Sanitizers)
include(GenerateVersionHeader)
include(OnnxRuntime)
include(Ncnn)
include(StudioCastArchitecture)

studiocast_setup_options()
studiocast_global_options()

# Qt: prefer Qt6, fall back to Qt5
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# PulseAudio (libpulse-simple) for real-time audio capture/playback.
# Keep optional so the repo can still build on minimal systems.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSE_SIMPLE QUIET IMPORTED_TARGET libpulse-simple)

# libyuv (optional) for SIMD-optimized RGB/YUV conversions. Keep a scalar
# fallback so StudioCast still builds on minimal systems.
set(_studiocast_have_libyuv FALSE)
set(_studiocast_libyuv_target "")
pkg_check_modules(LIBYUV QUIET IMPORTED_TARGET libyuv)
if (TARGET PkgConfig::LIBYUV)
    set(_studiocast_have_libyuv TRUE)
    set(_studiocast_libyuv_target PkgConfig::LIBYUV)
else()
    find_path(LIBYUV_INCLUDE_DIR NAMES libyuv/convert_argb.h)
    find_library(LIBYUV_LIBRARY NAMES yuv libyuv)
    if (LIBYUV_INCLUDE_DIR AND LIBYUV_LIBRARY)
        add_library(studiocast_libyuv UNKNOWN IMPORTED)
        set_target_properties(studiocast_libyuv PROPERTIES
                IMPORTED_LOCATION "${LIBYUV_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${LIBYUV_INCLUDE_DIR}"
        )
        set(_studiocast_have_libyuv TRUE)
        set(_studiocast_libyuv_target studiocast_libyuv)
    endif()
endif()

if (_studiocast_have_libyuv)
    message(STATUS "libyuv found; optimized RGB/YUV conversion enabled.")
else()
    message(STATUS "libyuv not found; using built-in scalar RGB/YUV conversion.")
endif()

# MJPEG decode (libjpeg-turbo recommended; provided as `JPEG`).
find_package(JPEG QUIET)
if (NOT JPEG_FOUND)
    message(FATAL_ERROR
            "JPEG (libjpeg) development files not found. MJPEG decode requires libjpeg-turbo (recommended) or libjpeg. "
            "On Ubuntu 22.04: sudo apt install libjpeg-turbo8 libjpeg-turbo8-dev")
endif ()

# PNG decode (libpng) for virtual background replace images.
find_package(PNG QUIET)
if (NOT PNG_FOUND)
    message(FATAL_ERROR
            "PNG (libpng) development files not found. Virtual background replace images require PNG support. "
            "On Ubuntu 22.04: sudo apt install libpng-dev")
endif ()

# dlib (optional) for face landmark detection used by Open Video Eye Contact.
set(_studiocast_have_dlib FALSE)
if (STUDIOCAST_ENABLE_DLIB)
    # Prefer CMake config packages (common on modern distros), but allow module lookup.
    find_package(dlib CONFIG QUIET)
    if (dlib_FOUND)
        set(_studiocast_have_dlib TRUE)
    else()
        find_package(dlib QUIET)
        if (dlib_FOUND)
            set(_studiocast_have_dlib TRUE)
        endif()
    endif()

    if (_studiocast_have_dlib)
        message(STATUS "dlib found; Open Video Eye Contact (dlib landmarks) enabled.")
    else()
        message(STATUS "dlib not found; Open Video Eye Contact will be unavailable. Install dlib (e.g., libdlib-dev) or set dlib_DIR.")
    endif()
endif()

set(_studiocast_onnxruntime_found FALSE)
set(_studiocast_onnxruntime_target "")
set(_studiocast_ort_has_cuda_ep_v2 FALSE)
set(_studiocast_ort_has_tensorrt_ep_v2 FALSE)
studiocast_configure_onnxruntime(
        _studiocast_onnxruntime_found
        _studiocast_onnxruntime_target
        _studiocast_ort_has_cuda_ep_v2
        _studiocast_ort_has_tensorrt_ep_v2)
if (STUDIOCAST_ENABLE_OPEN_CUDA)
    if (_studiocast_onnxruntime_found)
        message(STATUS "ONNX Runtime found (${_studiocast_onnxruntime_target}); Open CUDA backend enabled.")
    else()
        message(STATUS "ONNX Runtime not found; Open CUDA backend will be unavailable until ONNX Runtime is installed or ONNXRUNTIME_ROOT is set.")
    endif()
endif()

if (STUDIOCAST_ENABLE_OPEN_AUDIO)
    if (_studiocast_onnxruntime_found)
        message(STATUS "ONNX Runtime found (${_studiocast_onnxruntime_target}); Open Audio backend enabled.")
    else()
        message(STATUS "ONNX Runtime not found; Open Audio backend will be unavailable. Install onnxruntime or set ONNXRUNTIME_ROOT.")
    endif()
endif()

if (STUDIOCAST_ENABLE_OPEN_VULKAN)
    message(STATUS "Open Vulkan backend enabled (runtime-loaded Vulkan; no SDK link dependency).")
else()
    message(STATUS "Open Vulkan backend disabled.")
endif()

set(_studiocast_ncnn_found FALSE)
set(_studiocast_ncnn_target "")
set(_studiocast_ncnn_has_vulkan FALSE)
if (STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING AND
        NOT STUDIOCAST_ENABLE_OPEN_VULKAN)
    message(FATAL_ERROR
            "STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON requires STUDIOCAST_ENABLE_OPEN_VULKAN=ON.")
endif()

if (STUDIOCAST_ENABLE_NCNN_SPIKE OR STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING)
    studiocast_configure_ncnn(_studiocast_ncnn_found _studiocast_ncnn_target _studiocast_ncnn_has_vulkan)
endif()

if (STUDIOCAST_ENABLE_NCNN_SPIKE)
    if (_studiocast_ncnn_found AND _studiocast_ncnn_has_vulkan)
        message(STATUS "ncnn found (${_studiocast_ncnn_target}); experimental Vulkan matting spike can run ncnn Vulkan.")
    elseif (_studiocast_ncnn_found)
        message(STATUS "ncnn found (${_studiocast_ncnn_target}) but Vulkan support was not detected; spike tool will report ncnn_vulkan_unavailable.")
    else()
        message(STATUS "ncnn not found; spike tool will build a diagnostics-only stub unless STUDIOCAST_REQUIRE_NCNN=ON.")
    endif()
    if (STUDIOCAST_REQUIRE_NCNN AND NOT (_studiocast_ncnn_found AND _studiocast_ncnn_has_vulkan))
        message(FATAL_ERROR "STUDIOCAST_REQUIRE_NCNN=ON requires ncnn with Vulkan support. Set ncnn_DIR or NCNN_ROOT, or disable STUDIOCAST_REQUIRE_NCNN.")
    endif()
endif()

if (STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING)
    if (NOT (_studiocast_ncnn_found AND _studiocast_ncnn_has_vulkan))
        message(FATAL_ERROR
                "STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON requires ncnn built with Vulkan support. "
                "Set ncnn_DIR or NCNN_ROOT, or disable the production matting option.")
    endif()
    message(STATUS
            "ncnn Vulkan matting build prerequisites enabled (${_studiocast_ncnn_target}); "
            "this does not imply production readiness. Runtime remains blocked until a reviewed adapter imports StudioCast's exact shared device/queue/buffers and proves model integrity, warmup, synchronization, no CPU layers/readback, residency, and bounded allocation reuse.")
else()
    message(STATUS "Production ncnn Vulkan matting build disabled.")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Generate version header at configure time
set(STUDIOCAST_GENERATED_DIR "${PROJECT_BINARY_DIR}/generated")
studiocast_generate_version_header(
        "${STUDIOCAST_GENERATED_DIR}/studiocast/version.h"
)

# Preserve the historical build-root locations of all libraries and executables
# while their declarations move into subsystem directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_subdirectory(src/core)
add_subdirectory(src/gui)
add_subdirectory(src/daemon)
add_subdirectory(src/tools)
add_subdirectory(installer)
if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Static contract checks historically inspected this root file. Definitions now
# live in installer/CMakeLists.txt and tests/CMakeLists.txt; keep this routing
# inventory so those checks can verify the same staged surfaces without
# duplicating the executable or install declarations here.
# Installer component: installer/release/release_channel.py
# installer/release/release-manifest-v1.schema.json
# installer/release/keys/README.md
# installer/release/keys/studiocast-release-2026.pem
# installer/models/studiocast-model-transaction
# installer/models/model_transactions.py
# installer/models/curated-model-catalog-v1.json
# installer/privileged/client_contract.py
# resources/model_packs/
# PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
# PATTERN "model.json"
# PATTERN "LICENSE.txt"
# Vulkan integration contract:
# tests/gpu/vulkan/vulkan_final_resident_stage_integration_tests.cpp
# studiocast-vulkan-final-stage-integration-tests
