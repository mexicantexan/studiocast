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
studiocast_configure_onnxruntime(_studiocast_onnxruntime_found _studiocast_onnxruntime_target)
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
set(STUDIOCAST_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
studiocast_generate_version_header(
        "${STUDIOCAST_GENERATED_DIR}/studiocast/version.h"
)

# -----------------------------
# Core library (no Qt deps)
# -----------------------------
add_library(studiocast_core
        src/core/probe/probe.h
        src/core/probe/probe.cpp
        src/core/onnx/ort_session.h
        src/core/onnx/ort_session.cpp
        src/core/open_video/model_pack_registry.h
        src/core/open_video/model_pack_registry.cpp
        src/core/open_video/frame_analysis_cache.h
        src/core/open_video/latest_frame_wins_worker.h
        src/core/open_video/yunet_face_detector.h
        src/core/open_video/yunet_face_detector.cpp
        src/core/open_video/fastdvdnet_denoiser.h
        src/core/open_video/fastdvdnet_denoiser.cpp
        src/core/open_video/dlib_face_landmarks.h
        src/core/open_video/dlib_face_landmarks.cpp
        src/core/open_video/gaze_correction_eye_contact.h
        src/core/open_video/gaze_correction_eye_contact.cpp
        src/core/open_video/diagnostics.h
        src/core/open_video/diagnostics.cpp
        src/core/open_video/diagnose.h
        src/core/open_video/diagnose.cpp
        src/core/open_video/matting_session.h
        src/core/open_video/matting_session.cpp
        src/core/open_audio/open_audio_diagnostics.h
        src/core/open_audio/open_audio_diagnostics.cpp
        src/core/open_audio/open_audio_diagnose.h
        src/core/open_audio/open_audio_diagnose.cpp
        src/core/open_audio/model_pack_registry.h
        src/core/open_audio/model_pack_registry.cpp
        src/core/open_audio/open_audio_audio_processor.h
        src/core/open_audio/open_audio_audio_processor.cpp
        src/core/open_audio/open_audio_onnx_session.h
        src/core/open_audio/open_audio_onnx_session.cpp
        src/core/audio/effects/broadcast_audio_effect_contract.h
        src/core/util/dynlib.h
        src/core/util/dynlib.cpp
        src/core/util/json.h
        src/core/util/json.cpp
        src/core/util/exec.h
        src/core/util/exec.cpp
        src/core/util/fs.h
        src/core/util/fs.cpp
        src/core/util/os_release.h
        src/core/util/os_release.cpp
        src/core/util/strings.h
        src/core/util/strings.cpp
        src/core/util/proc.h
        src/core/util/proc.cpp
        src/core/util/xdg.h
        src/core/util/xdg.cpp
        src/core/config/daemon_config.h
        src/core/config/daemon_config.cpp
        src/core/config/settings.h
        src/core/config/settings.cpp
        src/core/maxine/gpu.h
        src/core/maxine/gpu.cpp
        src/core/maxine/gpu_selection.h
        src/core/maxine/gpu_selection.cpp
        src/core/maxine/availability.h
        src/core/maxine/availability.cpp
        src/core/maxine/reason_codes.h
        src/core/maxine/reason_codes.cpp
        src/core/maxine/paths.h
        src/core/maxine/paths.cpp
        src/core/maxine/maxine_manager.h
        src/core/maxine/maxine_manager.cpp
        src/core/maxine/nvcv_types.h
        src/core/maxine/nvcv_api.h
        src/core/maxine/nvcv_api.cpp
        src/core/maxine/vfx_api.h
        src/core/maxine/vfx_api.cpp
        src/core/maxine/afx_api.h
        src/core/maxine/afx_api.cpp
        src/core/maxine/afx/afx_effect.h
        src/core/maxine/afx/afx_effect.cpp
        src/core/maxine/afx/afx_audio_processor.h
        src/core/maxine/ar_api.h
        src/core/maxine/ar_api.cpp
        src/core/maxine/cuda_driver_api.h
        src/core/maxine/cuda_driver_api.cpp
        src/core/cuda/cuda_image.h
        src/core/cuda/cuda_image.cpp
        src/core/cuda/cuda_tensor.h
        src/core/cuda/cuda_tensor.cpp
        src/core/cuda/kernels/cuda_driver_cache.h
        src/core/cuda/kernels/ptx_pitch_contract.h
        src/core/cuda/kernels/resize_bilinear.h
        src/core/cuda/kernels/resize_bilinear_ptx.cpp
        src/core/cuda/kernels/preprocess_to_nchw.h
        src/core/cuda/kernels/preprocess_to_nchw_ptx.cpp
        src/core/cuda/kernels/open_cuda_vb_kernels.h
        src/core/cuda/kernels/open_cuda_vb_kernels_ptx.cpp
        src/core/maxine/cuda_crop_scale.h
        src/core/maxine/cuda_crop_scale.cpp
        src/core/maxine/cuda_vignette.h
        src/core/maxine/cuda_vignette.cpp
        src/core/maxine/effects/maxine_effect.h
        src/core/maxine/effects/maxine_effect.cpp
        src/core/maxine/effects/vfx_green_screen_effect.h
        src/core/maxine/effects/vfx_green_screen_effect.cpp
        src/core/maxine/effects/vfx_background_blur_effect.h
        src/core/maxine/effects/vfx_background_blur_effect.cpp
        src/core/maxine/effects/vfx_transfer_effect.h
        src/core/maxine/effects/vfx_transfer_effect.cpp
        src/core/maxine/effects/vfx_denoise_effect.h
        src/core/maxine/effects/vfx_denoise_effect.cpp
        src/core/maxine/effects/vfx_relighting_effect.h
        src/core/maxine/effects/vfx_relighting_effect.cpp
        src/core/maxine/effects/ar_eye_contact_effect.h
        src/core/maxine/effects/ar_eye_contact_effect.cpp
        src/core/maxine/effects/ar_auto_frame_tracker.h
        src/core/maxine/effects/ar_auto_frame_tracker.cpp
        src/core/audio/pulse/pactl.h
        src/core/audio/pulse/pactl.cpp
        src/core/audio/audio_consumer_detector.h
        src/core/audio/audio_consumer_detector.cpp
        src/core/audio/audio_device_safety.h
        src/core/audio/audio_device_safety.cpp
        src/core/audio/audio_backend_resolver.h
        src/core/audio/audio_backend_resolver.cpp
        src/core/audio/audio_processor.h
        src/core/audio/audio_processor.cpp
        src/core/audio/dsp/post_dsp_chain.h
        src/core/audio/dsp/post_dsp_chain.cpp
        src/core/audio/effects/broadcast_audio_effects.h
        src/core/audio/effects/broadcast_audio_effects_plan.h
        src/core/audio/effects/broadcast_audio_effects_plan.cpp
        src/core/audio/effects/broadcast_audio_effects_json.h
        src/core/audio/effects/broadcast_audio_effects_json.cpp
        src/core/audio/virtual_mic_state.h
        src/core/audio/virtual_mic_state.cpp
        src/core/audio/virtual_mic.h
        src/core/audio/virtual_mic.cpp
        src/core/audio/virtual_speaker_state.h
        src/core/audio/virtual_speaker_state.cpp
        src/core/audio/virtual_speaker.h
        src/core/audio/virtual_speaker.cpp
        src/core/audio/virtual_audio_service.h
        src/core/audio/virtual_audio_service.cpp
        src/core/video/v4l2loopback.h
        src/core/video/v4l2loopback.cpp
        src/core/video/v4l2_writer.h
        src/core/video/v4l2_writer.cpp
        src/core/video/gpu_frame.h
        src/core/video/pattern.h
        src/core/video/pattern.cpp
        src/core/video/image_ppm.h
        src/core/video/image_ppm.cpp
        src/core/video/feed.h
        src/core/video/feed.cpp
        src/core/video/convert.h
        src/core/video/convert.cpp
        src/core/video/convert_yuyv_rgb_internal.h
        src/core/video/convert_yuyv_rgb_scalar.cpp
        src/core/video/convert_yuyv_rgb_dispatch.cpp
        src/core/video/convert_rgb_yuyv_internal.h
        src/core/video/convert_rgb_yuyv_scalar.cpp
        src/core/video/convert_rgb_yuyv_dispatch.cpp
        src/core/video/convert_rgb_bgr_internal.h
        src/core/video/convert_rgb_bgr_scalar.cpp
        src/core/video/convert_rgb_bgr_dispatch.cpp
        src/core/video/scaling_policy.h
        src/core/video/scaling_policy.cpp
        src/core/video/mjpeg_decode.h
        src/core/video/mjpeg_decode.cpp
        src/core/video/effects/effect_types.h
        src/core/video/effects/effect_types.cpp
        src/core/video/effects/broadcast_effect_contract.h
        src/core/video/effects/broadcast_effect_maxine_gate.h
        src/core/video/effects/broadcast_effect_rules.h
        src/core/video/effects/broadcast_effect_rules.cpp
        src/core/video/effects/broadcast_effects.h
        src/core/video/effects/broadcast_effects.cpp
        src/core/video/effects/broadcast_effects_json.h
        src/core/video/effects/broadcast_effects_json.cpp
        src/core/video/effects/effect.h
        src/core/video/effects/effect_chain.h
        src/core/video/effects/effect_chain.cpp
        src/core/video/effects/effect_descriptors.h
        src/core/video/effects/effect_descriptors.cpp
        src/core/video/effects/mirror_effect.h
        src/core/video/effects/mirror_effect.cpp
        src/core/video/effects/focus_mask_plan.h
        src/core/video/effects/focus_mask_plan.cpp
        src/core/video/effects/box_blur.h
        src/core/video/effects/box_blur.cpp
        src/core/video/effects/background_blur_cpu.h
        src/core/video/effects/background_blur_cpu.cpp
        src/core/video/effects/background_remove_cpu.h
        src/core/video/effects/background_remove_cpu.cpp
        src/core/video/v4l2_capture.h
        src/core/video/v4l2_capture.cpp
        src/core/video/camera_pipeline.h
        src/core/video/camera_pipeline.cpp
        src/core/video/open_vulkan_matting_setup_policy.h
        src/core/video/open_vulkan_matting_setup_policy.cpp
        src/core/video/replace_background_cache_policy.h
        src/core/video/replace_background_cache_policy.cpp
        src/core/video/camera_effects_json.h
        src/core/video/camera_effects_json.cpp
        src/core/video/broadcast_camera_effects_json.h
        src/core/video/broadcast_camera_effects_json.cpp
        src/core/video/virtual_camera_service.h
        src/core/video/virtual_camera_service.cpp

        src/core/ipc/daemon_socket.h
        src/core/ipc/daemon_socket.cpp
        src/core/ipc/daemon_client.h
        src/core/ipc/daemon_client.cpp
        src/core/ipc/daemon_server.h
        src/core/ipc/daemon_server.cpp

)

set_target_properties(studiocast_core PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

set(_studiocast_x86_simd FALSE)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    set(_studiocast_x86_simd TRUE)
endif()

if (_studiocast_x86_simd)
    target_sources(studiocast_core PRIVATE
            src/core/video/convert_rgb_yuyv_ssse3.cpp
            src/core/video/convert_rgb_yuyv_avx2.cpp
            src/core/video/convert_yuyv_rgb_sse41.cpp
            src/core/video/convert_yuyv_rgb_avx2.cpp
            src/core/video/convert_rgb_bgr_ssse3.cpp
            src/core/video/convert_rgb_bgr_avx2.cpp
    )
    set_source_files_properties(src/core/video/convert_rgb_yuyv_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mssse3")
    set_source_files_properties(src/core/video/convert_rgb_yuyv_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    set_source_files_properties(src/core/video/convert_yuyv_rgb_sse41.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-msse4.1")
    set_source_files_properties(src/core/video/convert_yuyv_rgb_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    set_source_files_properties(src/core/video/convert_rgb_bgr_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mssse3")
    set_source_files_properties(src/core/video/convert_rgb_bgr_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_X86_SIMD=1)
else()
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_X86_SIMD=0)
endif()

if (STUDIOCAST_ENABLE_OPEN_VULKAN)
    target_sources(studiocast_core PRIVATE
            src/core/open_video/vulkan_matting_runtime.h
            src/core/open_video/vulkan_matting_runtime.cpp
            src/core/open_video/vulkan_matting_session.h
            src/core/open_video/vulkan_matting_session.cpp
            src/core/vulkan/vulkan_loader.h
            src/core/vulkan/vulkan_loader.cpp
            src/core/vulkan/vulkan_device.h
            src/core/vulkan/vulkan_device.cpp
            src/core/vulkan/vulkan_image.h
            src/core/vulkan/vulkan_image.cpp
            src/core/vulkan/vulkan_tensor.h
            src/core/vulkan/vulkan_tensor.cpp
            src/core/vulkan/frame_execution_plan.h
            src/core/vulkan/frame_execution_plan.cpp
            src/core/vulkan/kernels/resize_bilinear.h
            src/core/vulkan/kernels/resize_bilinear.cpp
            src/core/vulkan/kernels/utility_kernels.h
            src/core/vulkan/kernels/utility_kernels.cpp
            src/core/vulkan/kernels/shaders/resize_rgb24_bilinear_spv.h
            src/core/vulkan/kernels/shaders/utility_kernels_spv.h
            src/core/video/open_vulkan_mirror.h
            src/core/video/open_vulkan_mirror.cpp
            src/core/video/open_vulkan_auto_frame.h
            src/core/video/open_vulkan_auto_frame.cpp
            src/core/video/open_vulkan_virtual_background_blur.h
            src/core/video/open_vulkan_virtual_background_blur.cpp
            src/core/video/open_vulkan_virtual_background_remove.h
            src/core/video/open_vulkan_virtual_background_remove.cpp
            src/core/video/open_vulkan_vignette.h
            src/core/video/open_vulkan_vignette.cpp
    )
endif()

# Video color conversion is extremely hot (runs every frame). In Debug builds the lack of
# optimization can make the virtual camera feel laggy. Force optimization for this TU.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
                src/core/video/convert.cpp
                src/core/video/convert_yuyv_rgb_scalar.cpp
                src/core/video/convert_yuyv_rgb_dispatch.cpp
                src/core/video/convert_rgb_yuyv_scalar.cpp
                src/core/video/convert_rgb_yuyv_dispatch.cpp
                src/core/video/convert_rgb_bgr_scalar.cpp
                src/core/video/convert_rgb_bgr_dispatch.cpp
                PROPERTIES COMPILE_OPTIONS "-O3")
endif()

target_include_directories(studiocast_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

target_link_libraries(studiocast_core PUBLIC ${CMAKE_DL_LIBS})

if (_studiocast_have_libyuv)
    target_link_libraries(studiocast_core PRIVATE "${_studiocast_libyuv_target}")
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_LIBYUV=1)
else()
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_LIBYUV=0)
endif()

if (TARGET JPEG::JPEG)
    target_link_libraries(studiocast_core PUBLIC JPEG::JPEG)
else()
    target_include_directories(studiocast_core PUBLIC ${JPEG_INCLUDE_DIRS})
    target_link_libraries(studiocast_core PUBLIC ${JPEG_LIBRARIES})
endif()

if (TARGET PNG::PNG)
    target_link_libraries(studiocast_core PUBLIC PNG::PNG)
else()
    target_include_directories(studiocast_core PUBLIC ${PNG_INCLUDE_DIRS})
    target_link_libraries(studiocast_core PUBLIC ${PNG_LIBRARIES})
endif()

if (PULSE_SIMPLE_FOUND)
    target_sources(studiocast_core PRIVATE
            src/core/audio/audio_pipeline.h
            src/core/audio/audio_pipeline.cpp
    )
    target_link_libraries(studiocast_core PUBLIC PkgConfig::PULSE_SIMPLE)
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_PULSE_SIMPLE=1)
else()
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_PULSE_SIMPLE=0)
    message(STATUS "libpulse-simple not found via pkg-config; audio pipeline is disabled. Install libpulse-dev to enable.")
endif()

target_compile_definitions(studiocast_core PUBLIC
    STUDIOCAST_ENABLE_OPEN_CUDA=$<BOOL:${STUDIOCAST_ENABLE_OPEN_CUDA}>
    STUDIOCAST_ENABLE_OPEN_VULKAN=$<BOOL:${STUDIOCAST_ENABLE_OPEN_VULKAN}>
    STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=$<BOOL:${STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING}>
    STUDIOCAST_ENABLE_OPEN_AUDIO=$<BOOL:${STUDIOCAST_ENABLE_OPEN_AUDIO}>
    STUDIOCAST_HAVE_DLIB=$<BOOL:${_studiocast_have_dlib}>
)

if (STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING)
    target_link_libraries(studiocast_core PRIVATE "${_studiocast_ncnn_target}")
endif()

if (_studiocast_have_dlib)
    if (TARGET dlib::dlib)
        target_link_libraries(studiocast_core PUBLIC dlib::dlib)
    elseif (TARGET dlib)
        target_link_libraries(studiocast_core PUBLIC dlib)
    else()
        message(WARNING "dlib found but no CMake target was provided; linking may fail.")
    endif()
endif()

if ((_studiocast_onnxruntime_found) AND (STUDIOCAST_ENABLE_OPEN_CUDA OR STUDIOCAST_ENABLE_OPEN_AUDIO))
    target_link_libraries(studiocast_core PUBLIC ${_studiocast_onnxruntime_target})
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_ONNXRUNTIME=1)
else()
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_ONNXRUNTIME=0)
endif()

studiocast_set_project_warnings(studiocast_core)
studiocast_enable_sanitizers(studiocast_core)

# -----------------------------
# GUI executable
# -----------------------------
add_executable(studiocast
        src/main.cpp
        src/gui/theme.h
        src/gui/theme.cpp
        src/gui/text_edit_utils.h
        src/gui/text_edit_utils.cpp
        src/gui/resources/gui_resources.qrc
        src/gui/main_window.h
        src/gui/main_window.cpp
        src/gui/status/daemon_status_snapshot.h
        src/gui/status/daemon_status_snapshot.cpp
        src/gui/status/status_poller.h
        src/gui/status/status_poller.cpp
        src/gui/pages/home_page.h
        src/gui/pages/home_page.cpp
        src/gui/pages/engines_models_page.h
        src/gui/pages/engines_models_page.cpp
        src/gui/pages/support_page.h
        src/gui/pages/support_page.cpp
        src/gui/pages/settings_page.h
        src/gui/pages/settings_page.cpp
        src/gui/pages/advanced_page.h
        src/gui/pages/advanced_page.cpp
        src/gui/pages/audio_page.h
        src/gui/pages/audio_page.cpp
        src/gui/pages/video_page.h
        src/gui/pages/video_page.cpp
        src/core/probe/probe.h
)

target_link_libraries(studiocast PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        studiocast_core
)

target_include_directories(studiocast PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

target_compile_definitions(studiocast PRIVATE
        STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

studiocast_set_project_warnings(studiocast)
studiocast_enable_sanitizers(studiocast)

set_target_properties(studiocast PROPERTIES
        OUTPUT_NAME "studiocast"
)

# -----------------------------
# GUI installer executable
# -----------------------------
add_executable(studiocast-installer
        installer/gui/main.cpp
        installer/gui/installer_wizard.h
        installer/gui/installer_wizard.cpp
        src/gui/theme.h
        src/gui/theme.cpp
        src/gui/resources/gui_resources.qrc
)

target_link_libraries(studiocast-installer PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
)

target_include_directories(studiocast-installer PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/installer/gui"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

target_compile_definitions(studiocast-installer PRIVATE
        STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/installer/backend/studiocast-installer-backend"
        STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

studiocast_set_project_warnings(studiocast-installer)
studiocast_enable_sanitizers(studiocast-installer)

set_target_properties(studiocast-installer PROPERTIES
        OUTPUT_NAME "studiocast-installer"
)

# -----------------------------
# Probe tool 
# -----------------------------
add_executable(studiocast-probe
        src/tools/studiocast_probe_main.cpp
)

set_target_properties(studiocast-probe PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

target_link_libraries(studiocast-probe PRIVATE studiocast_core)

target_include_directories(studiocast-probe PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-probe)
studiocast_enable_sanitizers(studiocast-probe)

# -----------------------------
# Maxine helper tool 
# -----------------------------
add_executable(studiocast-maxine
        src/tools/studiocast_maxine_main.cpp
)

set_target_properties(studiocast-maxine PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_link_libraries(studiocast-maxine PRIVATE studiocast_core)

target_include_directories(studiocast-maxine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-maxine)
studiocast_enable_sanitizers(studiocast-maxine)

# -----------------------------
# Open CUDA helper tool
# -----------------------------
add_executable(studiocast-open
        src/tools/studiocast_open_main.cpp
)

set_target_properties(studiocast-open PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_link_libraries(studiocast-open PRIVATE studiocast_core)

target_include_directories(studiocast-open PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-open)
studiocast_enable_sanitizers(studiocast-open)

if (STUDIOCAST_ENABLE_NCNN_SPIKE)
    add_executable(studiocast-vulkan-matting-spike
            src/tools/studiocast_vulkan_matting_spike_main.cpp
    )

    set_target_properties(studiocast-vulkan-matting-spike PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    target_link_libraries(studiocast-vulkan-matting-spike PRIVATE studiocast_core)
    if (_studiocast_ncnn_found AND _studiocast_ncnn_has_vulkan)
        target_link_libraries(studiocast-vulkan-matting-spike PRIVATE "${_studiocast_ncnn_target}")
    endif()

    target_include_directories(studiocast-vulkan-matting-spike PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-vulkan-matting-spike PRIVATE
            STUDIOCAST_HAVE_NCNN=$<BOOL:${_studiocast_ncnn_found}>
            STUDIOCAST_NCNN_HAS_VULKAN=$<BOOL:${_studiocast_ncnn_has_vulkan}>
    )

    studiocast_set_project_warnings(studiocast-vulkan-matting-spike)
    studiocast_enable_sanitizers(studiocast-vulkan-matting-spike)
endif()

# -----------------------------
# Audio tool 
# -----------------------------
add_executable(studiocast-audio
        src/tools/studiocast_audio_main.cpp
)

target_link_libraries(studiocast-audio PRIVATE studiocast_core)

target_include_directories(studiocast-audio PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-audio)
studiocast_enable_sanitizers(studiocast-audio)

set_target_properties(studiocast-audio PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

# -----------------------------
# Video tool
# -----------------------------
add_executable(studiocast-video
        src/tools/studiocast_video_main.cpp
)

target_link_libraries(studiocast-video PRIVATE studiocast_core)

target_include_directories(studiocast-video PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-video)
studiocast_enable_sanitizers(studiocast-video)

set_target_properties(studiocast-video PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

if (STUDIOCAST_BUILD_BENCHMARKS)
    add_executable(studiocast-rgb-yuyv-bench
            src/tools/studiocast_rgb_yuyv_bench_main.cpp
    )

    target_link_libraries(studiocast-rgb-yuyv-bench PRIVATE studiocast_core)

    target_include_directories(studiocast-rgb-yuyv-bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-rgb-yuyv-bench)
    studiocast_enable_sanitizers(studiocast-rgb-yuyv-bench)

    set_target_properties(studiocast-rgb-yuyv-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_executable(studiocast-cuda-transfer-bench
            src/tools/studiocast_cuda_transfer_bench_main.cpp
    )

    target_link_libraries(studiocast-cuda-transfer-bench PRIVATE studiocast_core)

    target_include_directories(studiocast-cuda-transfer-bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-cuda-transfer-bench)
    studiocast_enable_sanitizers(studiocast-cuda-transfer-bench)

    set_target_properties(studiocast-cuda-transfer-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_executable(studiocast-resize-backend-bench
            src/tools/studiocast_resize_backend_bench_main.cpp
    )

    target_link_libraries(studiocast-resize-backend-bench PRIVATE studiocast_core)

    target_include_directories(studiocast-resize-backend-bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-resize-backend-bench)
    studiocast_enable_sanitizers(studiocast-resize-backend-bench)

    set_target_properties(studiocast-resize-backend-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
endif()


# -----------------------------
# Daemon control CLI (studiocastctl)
# -----------------------------
add_executable(studiocastctl
        src/tools/studiocastctl_main.cpp
)

target_link_libraries(studiocastctl PRIVATE studiocast_core)

target_include_directories(studiocastctl PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocastctl)
studiocast_enable_sanitizers(studiocastctl)

set_target_properties(studiocastctl PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
        OUTPUT_NAME "studiocastctl"
)


# -----------------------------
# Background daemon
# -----------------------------
add_executable(studiocastd
        src/daemon/studiocastd_main.cpp
)

target_link_libraries(studiocastd PRIVATE studiocast_core)

target_include_directories(studiocastd PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocastd)
studiocast_enable_sanitizers(studiocastd)

set_target_properties(studiocastd PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
        OUTPUT_NAME "studiocastd"
)


# -----------------------------
# Install step
# -----------------------------
include(GNUInstallDirs)
install(TARGETS studiocast studiocast-probe studiocast-maxine studiocast-open studiocast-audio studiocast-video studiocastctl
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Runtime
)

install(TARGETS studiocast-installer
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Installer
)

install(TARGETS studiocastd
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Runtime
)

install(PROGRAMS installer/backend/studiocast-installer-backend
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer"
        COMPONENT Installer
)

install(FILES
        installer/privileged/__init__.py
        installer/privileged/client_contract.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/privileged"
        COMPONENT Installer
)

install(FILES
        packaging/release/__init__.py
        packaging/release/release_channel.py
        packaging/release/release-manifest-v1.schema.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/release"
        COMPONENT Installer
)

# Curated model transactions execute from the trusted Installer component, not
# from a target-built or user-selected source tree. Keep the installed relative
# layout expected by model_transactions.py while staging metadata only.
install(PROGRAMS installer/models/studiocast-model-transaction
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/models"
        COMPONENT Installer
)

install(FILES
        installer/models/__init__.py
        installer/models/model_transactions.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/models"
        COMPONENT Installer
)

install(FILES packaging/models/curated-model-catalog-v1.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/packaging/models"
        COMPONENT Installer
)

install(DIRECTORY resources/model_packs/
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/resources/model_packs"
        COMPONENT Installer
        FILES_MATCHING
        PATTERN "model.json"
        PATTERN "LICENSE.txt"
)

# Production release engineering installs <key-id>.pem trust roots here.  No
# production key is committed yet, so stable-channel verification fails closed.
install(FILES packaging/release/keys/README.md
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/trust/keys"
        COMPONENT Installer
)

if (BUILD_TESTING AND PULSE_SIMPLE_FOUND)
    add_executable(studiocast-audio-tests
            tests/audio_service_tests.cpp
    )

    target_link_libraries(studiocast-audio-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-audio-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-audio-tests)
    studiocast_enable_sanitizers(studiocast-audio-tests)

    set_target_properties(studiocast-audio-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-audio-tests COMMAND studiocast-audio-tests)
elseif (BUILD_TESTING)
    message(STATUS "Skipping studiocast-audio-tests because libpulse-simple was not found.")
endif()

if (BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME studiocast-embedded-ptx-freshness
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/dev/validate_embedded_ptx.py"
                        --source-root "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(studiocast-embedded-ptx-freshness PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        add_test(NAME studiocast-embedded-vulkan-spv-freshness
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/dev/validate_embedded_vulkan_spv.py"
                        --source-root "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(studiocast-embedded-vulkan-spv-freshness PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    else()
        message(STATUS "Skipping embedded shader freshness tests because Python3 was not found.")
    endif()

    add_executable(studiocast-maxine-paths-tests
            tests/maxine_paths_tests.cpp
    )

    target_link_libraries(studiocast-maxine-paths-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-maxine-paths-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-maxine-paths-tests PRIVATE
            STUDIOCAST_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    )

    studiocast_set_project_warnings(studiocast-maxine-paths-tests)
    studiocast_enable_sanitizers(studiocast-maxine-paths-tests)

    set_target_properties(studiocast-maxine-paths-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-maxine-paths-tests COMMAND studiocast-maxine-paths-tests)

    add_executable(studiocast-nvcv-validation-tests
            tests/nvcv_validation_tests.cpp
    )

    target_link_libraries(studiocast-nvcv-validation-tests PRIVATE
            studiocast_core
    )

    target_include_directories(studiocast-nvcv-validation-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-nvcv-validation-tests)
    studiocast_enable_sanitizers(studiocast-nvcv-validation-tests)

    set_target_properties(studiocast-nvcv-validation-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-nvcv-validation-tests
            COMMAND studiocast-nvcv-validation-tests)

    add_executable(studiocast-model-registry-tests
            tests/model_pack_registry_tests.cpp
    )

    target_link_libraries(studiocast-model-registry-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-model-registry-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-model-registry-tests)
    studiocast_enable_sanitizers(studiocast-model-registry-tests)

    set_target_properties(studiocast-model-registry-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-model-registry-tests COMMAND studiocast-model-registry-tests)
    set_tests_properties(studiocast-model-registry-tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_executable(studiocast-cuda-driver-api-tests
            tests/cuda_driver_api_tests.cpp
    )

    target_link_libraries(studiocast-cuda-driver-api-tests PRIVATE
            studiocast_core
    )

    target_include_directories(studiocast-cuda-driver-api-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-cuda-driver-api-tests)
    studiocast_enable_sanitizers(studiocast-cuda-driver-api-tests)

    set_target_properties(studiocast-cuda-driver-api-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-cuda-driver-api-tests
            COMMAND studiocast-cuda-driver-api-tests)

    add_executable(studiocast-ipc-tests
            tests/ipc_client_tests.cpp
    )

    target_link_libraries(studiocast-ipc-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-ipc-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-ipc-tests)
    studiocast_enable_sanitizers(studiocast-ipc-tests)

    set_target_properties(studiocast-ipc-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-ipc-tests COMMAND studiocast-ipc-tests)

    add_executable(studiocast-gui-status-tests
            tests/gui_status_snapshot_tests.cpp
            src/gui/text_edit_utils.h
            src/gui/text_edit_utils.cpp
            src/gui/pages/engines_models_page.h
            src/gui/pages/engines_models_page.cpp
            src/gui/pages/video_page.h
            src/gui/pages/video_page.cpp
            src/gui/status/daemon_status_snapshot.h
            src/gui/status/daemon_status_snapshot.cpp
            src/gui/status/pending_daemon_write_guard.h
            src/gui/status/status_poller.h
            src/gui/status/status_poller.cpp
    )

    target_link_libraries(studiocast-gui-status-tests PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            studiocast_core
    )

    target_include_directories(studiocast-gui-status-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-gui-status-tests)
    studiocast_enable_sanitizers(studiocast-gui-status-tests)

    set_target_properties(studiocast-gui-status-tests PROPERTIES
            AUTOMOC ON
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-gui-status-tests COMMAND studiocast-gui-status-tests)

    add_executable(studiocast-effect-schema-tests
            tests/effect_schema_tests.cpp
    )

    target_link_libraries(studiocast-effect-schema-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-effect-schema-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-effect-schema-tests)
    studiocast_enable_sanitizers(studiocast-effect-schema-tests)

    set_target_properties(studiocast-effect-schema-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-effect-schema-tests COMMAND studiocast-effect-schema-tests)

    add_executable(studiocast-daemon-status-tests
            tests/daemon_status_tests.cpp
    )

    target_link_libraries(studiocast-daemon-status-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-daemon-status-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-daemon-status-tests)
    studiocast_enable_sanitizers(studiocast-daemon-status-tests)

    set_target_properties(studiocast-daemon-status-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-daemon-status-tests
            COMMAND studiocast-daemon-status-tests)

    add_executable(studiocast-exec-tests
            tests/exec_tests.cpp
    )

    target_link_libraries(studiocast-exec-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-exec-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-exec-tests)
    studiocast_enable_sanitizers(studiocast-exec-tests)

    set_target_properties(studiocast-exec-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-exec-tests COMMAND studiocast-exec-tests)

    add_executable(studiocast-installer-backend-tests
            tests/installer_backend_tests.cpp
    )

    target_link_libraries(studiocast-installer-backend-tests PRIVATE
            studiocast_core
    )

    target_include_directories(studiocast-installer-backend-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-installer-backend-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    studiocast_set_project_warnings(studiocast-installer-backend-tests)
    studiocast_enable_sanitizers(studiocast-installer-backend-tests)

    set_target_properties(studiocast-installer-backend-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-installer-backend-tests
            COMMAND studiocast-installer-backend-tests)

    if (Python3_Interpreter_FOUND)
        add_test(NAME studiocast-installer-backend-core-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/installer_backend_core_tests.py")
        set_tests_properties(studiocast-installer-backend-core-tests PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME studiocast-installer-packaging-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/installer_packaging_tests.py")
        set_tests_properties(studiocast-installer-packaging-tests PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    add_executable(studiocast-installer-wizard-tests
            tests/installer_wizard_tests.cpp
            installer/gui/installer_wizard.h
            installer/gui/installer_wizard.cpp
    )

    target_link_libraries(studiocast-installer-wizard-tests PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
    )

    target_include_directories(studiocast-installer-wizard-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/installer/gui"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-installer-wizard-tests PRIVATE
            STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/installer/backend/studiocast-installer-backend"
            STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    studiocast_set_project_warnings(studiocast-installer-wizard-tests)
    studiocast_enable_sanitizers(studiocast-installer-wizard-tests)

    set_target_properties(studiocast-installer-wizard-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-installer-wizard-tests
            COMMAND studiocast-installer-wizard-tests)

    add_executable(studiocast-cmake-open-backend-cache-tests
            tests/cmake_open_backend_cache_tests.cpp
    )

    target_link_libraries(studiocast-cmake-open-backend-cache-tests PRIVATE
            studiocast_core
    )

    target_include_directories(studiocast-cmake-open-backend-cache-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-cmake-open-backend-cache-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            STUDIOCAST_CMAKE_COMMAND="${CMAKE_COMMAND}"
    )

    studiocast_set_project_warnings(studiocast-cmake-open-backend-cache-tests)
    studiocast_enable_sanitizers(studiocast-cmake-open-backend-cache-tests)

    set_target_properties(studiocast-cmake-open-backend-cache-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-cmake-open-backend-cache-tests
            COMMAND studiocast-cmake-open-backend-cache-tests)

    add_executable(studiocast-video-tests
            tests/video_service_tests.cpp
            tests/latest_frame_wins_worker_tests.cpp
            tests/open_video_denoise_contract_tests.cpp
            tests/frame_artifact_cache_tests.cpp
            tests/cuda_kernel_cache_tests.cpp
            tests/open_vulkan_matting_setup_policy_tests.cpp
            tests/replace_background_cache_policy_tests.cpp
            tests/v4l2_capture_negotiation_tests.cpp
            tests/video_effect_cpu_tests.cpp
            tests/rgb_yuyv_convert_tests.cpp
    )

    target_link_libraries(studiocast-video-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-video-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-video-tests)
    studiocast_enable_sanitizers(studiocast-video-tests)

    set_target_properties(studiocast-video-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-video-tests COMMAND studiocast-video-tests)

    add_executable(studiocast-vulkan-frame-plan-tests
            tests/vulkan_frame_execution_plan_tests.cpp
            src/core/vulkan/frame_execution_plan.cpp
    )

    target_include_directories(studiocast-vulkan-frame-plan-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(studiocast-vulkan-frame-plan-tests)
    studiocast_enable_sanitizers(studiocast-vulkan-frame-plan-tests)

    set_target_properties(studiocast-vulkan-frame-plan-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-vulkan-frame-plan-tests
            COMMAND studiocast-vulkan-frame-plan-tests)

    add_executable(studiocast-vulkan-capability-audit-tests
            tests/vulkan_capability_audit_tests.cpp
    )

    target_include_directories(studiocast-vulkan-capability-audit-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    target_compile_definitions(studiocast-vulkan-capability-audit-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    studiocast_set_project_warnings(studiocast-vulkan-capability-audit-tests)
    studiocast_enable_sanitizers(studiocast-vulkan-capability-audit-tests)

    set_target_properties(studiocast-vulkan-capability-audit-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-vulkan-capability-audit-tests
            COMMAND studiocast-vulkan-capability-audit-tests)

    if (STUDIOCAST_ENABLE_OPEN_VULKAN)
        add_executable(studiocast-vulkan-kernel-tests
                tests/vulkan_kernel_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-kernel-tests PRIVATE
                studiocast_core
        )

        target_include_directories(studiocast-vulkan-kernel-tests PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${STUDIOCAST_GENERATED_DIR}"
        )

        studiocast_set_project_warnings(studiocast-vulkan-kernel-tests)
        studiocast_enable_sanitizers(studiocast-vulkan-kernel-tests)

        set_target_properties(studiocast-vulkan-kernel-tests PROPERTIES
                AUTOMOC OFF
                AUTOUIC OFF
                AUTORCC OFF
        )

        add_test(NAME studiocast-vulkan-kernel-tests
                COMMAND studiocast-vulkan-kernel-tests)

        add_executable(studiocast-vulkan-matting-runtime-tests
                tests/vulkan_matting_runtime_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-matting-runtime-tests PRIVATE
                studiocast_core
        )

        target_include_directories(studiocast-vulkan-matting-runtime-tests PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${STUDIOCAST_GENERATED_DIR}"
        )

        studiocast_set_project_warnings(studiocast-vulkan-matting-runtime-tests)
        studiocast_enable_sanitizers(studiocast-vulkan-matting-runtime-tests)

        set_target_properties(studiocast-vulkan-matting-runtime-tests PROPERTIES
                AUTOMOC OFF
                AUTOUIC OFF
                AUTORCC OFF
        )

        add_test(NAME studiocast-vulkan-matting-runtime-tests
                COMMAND studiocast-vulkan-matting-runtime-tests)
    endif()

    add_executable(studiocast-v4l2loopback-diagnostics-tests
            tests/v4l2loopback_diagnostics_tests.cpp
    )

    target_link_libraries(studiocast-v4l2loopback-diagnostics-tests PRIVATE studiocast_core)

    target_include_directories(studiocast-v4l2loopback-diagnostics-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-v4l2loopback-diagnostics-tests)
    studiocast_enable_sanitizers(studiocast-v4l2loopback-diagnostics-tests)

    set_target_properties(studiocast-v4l2loopback-diagnostics-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-v4l2loopback-diagnostics-tests
            COMMAND studiocast-v4l2loopback-diagnostics-tests)
endif()
