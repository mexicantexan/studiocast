# -----------------------------
# Neutral libraries and high-level compatibility core
# -----------------------------
set(_studiocast_util_sources
        ${PROJECT_SOURCE_DIR}/src/core/util/dynlib.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/exec.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/fs.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/json.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/os_release.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/proc.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/strings.cpp
        ${PROJECT_SOURCE_DIR}/src/core/util/xdg.cpp)

set(_studiocast_contract_sources
        ${PROJECT_SOURCE_DIR}/src/core/audio/audio_backend_resolver.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/audio_processor.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/effects/broadcast_audio_effects_json.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/effects/broadcast_audio_effects_plan.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/reason_codes.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/broadcast_camera_effects_json.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/camera_effects_json.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/broadcast_effect_rules.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/broadcast_effects.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/broadcast_effects_json.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/effect_descriptors.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/effect_types.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/video_config_types.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/frame_execution_plan.cpp)

set(_studiocast_config_sources
        ${PROJECT_SOURCE_DIR}/src/core/config/daemon_config.cpp
        ${PROJECT_SOURCE_DIR}/src/core/config/settings.cpp)

set(_studiocast_ipc_sources
        ${PROJECT_SOURCE_DIR}/src/core/ipc/daemon_client.cpp
        ${PROJECT_SOURCE_DIR}/src/core/ipc/daemon_server.cpp
        ${PROJECT_SOURCE_DIR}/src/core/ipc/daemon_socket.cpp)

set(_studiocast_cuda_support_sources
        ${PROJECT_SOURCE_DIR}/src/core/maxine/cuda_driver_api.cpp
        ${PROJECT_SOURCE_DIR}/src/core/cuda/cuda_image.cpp
        ${PROJECT_SOURCE_DIR}/src/core/cuda/cuda_tensor.cpp
        ${PROJECT_SOURCE_DIR}/src/core/cuda/kernels/open_cuda_vb_kernels_ptx.cpp
        ${PROJECT_SOURCE_DIR}/src/core/cuda/kernels/preprocess_to_nchw_ptx.cpp
        ${PROJECT_SOURCE_DIR}/src/core/cuda/kernels/resize_bilinear_ptx.cpp)

set(_studiocast_core_base_sources
        ${PROJECT_SOURCE_DIR}/src/core/audio/audio_consumer_detector.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/audio_device_safety.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/dsp/post_dsp_chain.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/pulse/pactl.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/virtual_audio_service.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/virtual_mic.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/virtual_mic_state.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/virtual_speaker.cpp
        ${PROJECT_SOURCE_DIR}/src/core/audio/virtual_speaker_state.cpp
        ${PROJECT_SOURCE_DIR}/src/core/config/daemon_service_config_adapters.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/afx/afx_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/afx_api.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/ar_api.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/availability.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/cuda_crop_scale.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/cuda_vignette.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/ar_auto_frame_tracker.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/ar_eye_contact_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/maxine_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/vfx_background_blur_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/vfx_denoise_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/vfx_green_screen_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/vfx_relighting_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/effects/vfx_transfer_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/gpu.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/gpu_selection.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/maxine_manager.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/nvcv_api.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/paths.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/production_resident_frame_executor.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/resident_frame_section.cpp
        ${PROJECT_SOURCE_DIR}/src/core/maxine/vfx_api.cpp
        ${PROJECT_SOURCE_DIR}/src/core/onnx/ort_session.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_audio/model_pack_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_audio/open_audio_audio_processor.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_audio/open_audio_diagnose.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_audio/open_audio_diagnostics.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_audio/open_audio_onnx_session.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/diagnose.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/diagnostics.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/dlib_face_landmarks.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/fastdvdnet_denoiser.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/gaze_correction_eye_contact.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/matting_session.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/model_pack_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/yunet_face_detector.cpp
        ${PROJECT_SOURCE_DIR}/src/core/probe/probe.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/camera_pipeline.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_dispatch.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_scalar.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_dispatch.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_scalar.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_dispatch.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_scalar.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/background_blur_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/background_remove_cpu.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/box_blur.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/effect_chain.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/focus_mask_plan.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/effects/mirror_effect.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/feed.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/image_ppm.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/mjpeg_decode.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_eye_contact.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_matting_setup_policy.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_video_noise_removal.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/pattern.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/replace_background_cache_policy.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/scaling_policy.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/v4l2_capture.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/v4l2_writer.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/v4l2loopback.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/virtual_camera_service.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/yuyv_passthrough.cpp)

set(_studiocast_core_x86_sources
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_ssse3.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_avx2.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_sse41.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_avx2.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_ssse3.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_avx2.cpp)

set(_studiocast_core_vulkan_sources
        ${PROJECT_SOURCE_DIR}/src/core/open_video/vulkan_matting_runtime.cpp
        ${PROJECT_SOURCE_DIR}/src/core/open_video/vulkan_matting_session.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/vulkan_loader.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/vulkan_device.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/vulkan_image.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/vulkan_tensor.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/kernels/resize_bilinear.cpp
        ${PROJECT_SOURCE_DIR}/src/core/vulkan/kernels/utility_kernels.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_mirror.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_final_resident_stage.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_auto_frame.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_virtual_background_blur.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_virtual_background_remove.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_virtual_background_replace.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_virtual_key_light.cpp
        ${PROJECT_SOURCE_DIR}/src/core/video/open_vulkan_vignette.cpp)

add_library(studiocast_util STATIC ${_studiocast_util_sources})
add_library(studiocast_contracts STATIC ${_studiocast_contract_sources})
add_library(studiocast_config STATIC ${_studiocast_config_sources})
add_library(studiocast_ipc STATIC ${_studiocast_ipc_sources})
add_library(studiocast_cuda_support STATIC ${_studiocast_cuda_support_sources})
add_library(studiocast_core STATIC ${_studiocast_core_base_sources})

foreach (target IN ITEMS
        studiocast_util studiocast_contracts studiocast_config
        studiocast_ipc studiocast_cuda_support studiocast_core)
    set_target_properties(${target} PROPERTIES
            AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    target_include_directories(${target} PUBLIC
            "${PROJECT_SOURCE_DIR}/src")
    studiocast_set_project_warnings(${target})
    studiocast_enable_sanitizers(${target})
endforeach ()

target_include_directories(studiocast_core PUBLIC "${STUDIOCAST_GENERATED_DIR}")
target_link_libraries(studiocast_util PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(studiocast_contracts PUBLIC studiocast_util)
target_link_libraries(studiocast_config PUBLIC studiocast_contracts PRIVATE studiocast_util)
target_link_libraries(studiocast_ipc PRIVATE studiocast_util)
target_link_libraries(studiocast_cuda_support PUBLIC studiocast_util)
target_link_libraries(studiocast_core PUBLIC
        studiocast_contracts
        studiocast_config
        studiocast_ipc
        studiocast_cuda_support
        studiocast_util)

set(_studiocast_x86_simd FALSE)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    set(_studiocast_x86_simd TRUE)
endif()

if (_studiocast_x86_simd)
    target_sources(studiocast_core PRIVATE ${_studiocast_core_x86_sources})
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mssse3")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_sse41.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-msse4.1")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_ssse3.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mssse3")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-O3;-mavx2")
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_X86_SIMD=1)
else()
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_X86_SIMD=0)
endif()

if (STUDIOCAST_ENABLE_OPEN_VULKAN)
    target_sources(studiocast_core PRIVATE ${_studiocast_core_vulkan_sources})
endif()

# Video color conversion is extremely hot (runs every frame). In Debug builds the lack of
# optimization can make the virtual camera feel laggy. Force optimization for this TU.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
                ${PROJECT_SOURCE_DIR}/src/core/video/convert.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_scalar.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_yuyv_rgb_dispatch.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_scalar.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_yuyv_dispatch.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_scalar.cpp
                ${PROJECT_SOURCE_DIR}/src/core/video/convert_rgb_bgr_dispatch.cpp
                PROPERTIES COMPILE_OPTIONS "-O3")
endif()

if (_studiocast_have_libyuv)
    target_link_libraries(studiocast_core PRIVATE "${_studiocast_libyuv_target}")
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_LIBYUV=1)
else()
    target_compile_definitions(studiocast_core PRIVATE STUDIOCAST_HAVE_LIBYUV=0)
endif()

if (TARGET JPEG::JPEG)
    target_link_libraries(studiocast_core PRIVATE JPEG::JPEG)
else()
    target_include_directories(studiocast_core PRIVATE ${JPEG_INCLUDE_DIRS})
    target_link_libraries(studiocast_core PRIVATE ${JPEG_LIBRARIES})
endif()

if (TARGET PNG::PNG)
    target_link_libraries(studiocast_core PRIVATE PNG::PNG)
else()
    target_include_directories(studiocast_core PRIVATE ${PNG_INCLUDE_DIRS})
    target_link_libraries(studiocast_core PRIVATE ${PNG_LIBRARIES})
endif()

if (PULSE_SIMPLE_FOUND)
    target_sources(studiocast_core PRIVATE
            ${PROJECT_SOURCE_DIR}/src/core/audio/audio_pipeline.h
            ${PROJECT_SOURCE_DIR}/src/core/audio/audio_pipeline.cpp
    )
    target_link_libraries(studiocast_core PRIVATE PkgConfig::PULSE_SIMPLE)
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
target_compile_definitions(studiocast_core PRIVATE
    STUDIOCAST_ORT_HAS_CUDA_EP_V2=$<BOOL:${_studiocast_ort_has_cuda_ep_v2}>
    STUDIOCAST_ORT_HAS_TENSORRT_EP_V2=$<BOOL:${_studiocast_ort_has_tensorrt_ep_v2}>
)

if (STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING)
    target_link_libraries(studiocast_core PRIVATE "${_studiocast_ncnn_target}")
endif()

if (_studiocast_have_dlib)
    if (TARGET dlib::dlib)
        target_link_libraries(studiocast_core PRIVATE dlib::dlib)
    elseif (TARGET dlib)
        target_link_libraries(studiocast_core PRIVATE dlib)
    else()
        message(WARNING "dlib found but no CMake target was provided; linking may fail.")
    endif()
endif()

if ((_studiocast_onnxruntime_found) AND (STUDIOCAST_ENABLE_OPEN_CUDA OR STUDIOCAST_ENABLE_OPEN_AUDIO))
    target_link_libraries(studiocast_core PRIVATE ${_studiocast_onnxruntime_target})
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_ONNXRUNTIME=1)
else()
    target_compile_definitions(studiocast_core PUBLIC STUDIOCAST_HAVE_ONNXRUNTIME=0)
endif()


studiocast_assert_direct_dependencies(studiocast_util)
studiocast_assert_direct_dependencies(studiocast_contracts
        PUBLIC_STUDIOCAST_DEPS studiocast_util)
studiocast_assert_direct_dependencies(studiocast_config
        PUBLIC_STUDIOCAST_DEPS studiocast_contracts
        PRIVATE_STUDIOCAST_DEPS studiocast_util)
studiocast_assert_direct_dependencies(studiocast_ipc
        PRIVATE_STUDIOCAST_DEPS studiocast_util)
studiocast_assert_direct_dependencies(studiocast_cuda_support
        PUBLIC_STUDIOCAST_DEPS studiocast_util)
studiocast_assert_direct_dependencies(studiocast_core
        PUBLIC_STUDIOCAST_DEPS
        studiocast_contracts studiocast_config studiocast_ipc
        studiocast_cuda_support studiocast_util)

foreach (target IN ITEMS
        studiocast_util studiocast_contracts studiocast_config
        studiocast_ipc studiocast_cuda_support)
    studiocast_assert_neutral_target(${target})
endforeach ()

set(_studiocast_expected_sources
        ${_studiocast_util_sources}
        ${_studiocast_contract_sources}
        ${_studiocast_config_sources}
        ${_studiocast_ipc_sources}
        ${_studiocast_cuda_support_sources}
        ${_studiocast_core_base_sources})
set(_studiocast_expected_source_count 108)
if (_studiocast_x86_simd)
    list(APPEND _studiocast_expected_sources ${_studiocast_core_x86_sources})
    math(EXPR _studiocast_expected_source_count "${_studiocast_expected_source_count} + 6")
endif ()
if (STUDIOCAST_ENABLE_OPEN_VULKAN)
    list(APPEND _studiocast_expected_sources ${_studiocast_core_vulkan_sources})
    math(EXPR _studiocast_expected_source_count "${_studiocast_expected_source_count} + 16")
endif ()
if (PULSE_SIMPLE_FOUND)
    list(APPEND _studiocast_expected_sources ${PROJECT_SOURCE_DIR}/src/core/audio/audio_pipeline.cpp)
    math(EXPR _studiocast_expected_source_count "${_studiocast_expected_source_count} + 1")
endif ()
studiocast_assert_cpp_source_ownership(
        TARGETS
        studiocast_util studiocast_contracts studiocast_config
        studiocast_ipc studiocast_cuda_support studiocast_core
        EXPECTED_COUNT ${_studiocast_expected_source_count}
        EXPECTED_SOURCES ${_studiocast_expected_sources})
