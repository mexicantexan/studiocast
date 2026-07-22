# -----------------------------
# Probe tool
# -----------------------------
add_executable(studiocast-probe
        ${PROJECT_SOURCE_DIR}/src/tools/studiocast_probe_main.cpp
)

set_target_properties(studiocast-probe PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

target_link_libraries(studiocast-probe PRIVATE studiocast_runtime_config)

target_include_directories(studiocast-probe PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-probe)
studiocast_enable_sanitizers(studiocast-probe)

# -----------------------------
# Maxine helper tool
# -----------------------------
add_executable(studiocast-maxine
        ${PROJECT_SOURCE_DIR}/src/tools/studiocast_maxine_main.cpp
)

set_target_properties(studiocast-maxine PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_link_libraries(studiocast-maxine PRIVATE studiocast_video_runtime)

target_include_directories(studiocast-maxine PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-maxine)
studiocast_enable_sanitizers(studiocast-maxine)

# -----------------------------
# Open CUDA helper tool
# -----------------------------
add_executable(studiocast-open
        ${PROJECT_SOURCE_DIR}/src/tools/studiocast_open_main.cpp
)

set_target_properties(studiocast-open PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

target_link_libraries(studiocast-open PRIVATE
        studiocast_open_audio_backend
        studiocast_open_video_backend)

target_include_directories(studiocast-open PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

studiocast_set_project_warnings(studiocast-open)
studiocast_enable_sanitizers(studiocast-open)

if (STUDIOCAST_ENABLE_NCNN_SPIKE)
    add_executable(studiocast-vulkan-matting-spike
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_vulkan_matting_spike_main.cpp
    )

    set_target_properties(studiocast-vulkan-matting-spike PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    target_link_libraries(studiocast-vulkan-matting-spike PRIVATE
            studiocast_onnx_support
            studiocast_vulkan_backend)
    if (_studiocast_ncnn_found AND _studiocast_ncnn_has_vulkan)
        target_link_libraries(studiocast-vulkan-matting-spike PRIVATE "${_studiocast_ncnn_target}")
    endif()

    target_include_directories(studiocast-vulkan-matting-spike PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
        ${PROJECT_SOURCE_DIR}/src/tools/studiocast_audio_main.cpp
)

target_link_libraries(studiocast-audio PRIVATE studiocast_audio_runtime)

target_include_directories(studiocast-audio PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
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
        ${PROJECT_SOURCE_DIR}/src/tools/studiocast_video_main.cpp
)

target_link_libraries(studiocast-video PRIVATE studiocast_video_io)

target_include_directories(studiocast-video PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
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
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_rgb_yuyv_bench_main.cpp
    )

    target_link_libraries(studiocast-rgb-yuyv-bench PRIVATE
            studiocast_video_io)

    target_include_directories(studiocast-rgb-yuyv-bench PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-rgb-yuyv-bench)
    studiocast_enable_sanitizers(studiocast-rgb-yuyv-bench)

    set_target_properties(studiocast-rgb-yuyv-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_executable(studiocast-v4l2-writer-bench
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_v4l2_writer_bench_main.cpp
    )

    target_link_libraries(studiocast-v4l2-writer-bench PRIVATE
            studiocast_video_io)

    target_include_directories(studiocast-v4l2-writer-bench PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-v4l2-writer-bench)
    studiocast_enable_sanitizers(studiocast-v4l2-writer-bench)

    set_target_properties(studiocast-v4l2-writer-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_executable(studiocast-cuda-transfer-bench
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_cuda_transfer_bench_main.cpp
    )

    target_link_libraries(studiocast-cuda-transfer-bench PRIVATE
            studiocast_cuda_support)

    target_include_directories(studiocast-cuda-transfer-bench PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_resize_backend_bench_main.cpp
    )

    target_link_libraries(studiocast-resize-backend-bench PRIVATE
            studiocast_cuda_support
            studiocast_video_support
            studiocast_vulkan_backend)

    target_include_directories(studiocast-resize-backend-bench PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-resize-backend-bench)
    studiocast_enable_sanitizers(studiocast-resize-backend-bench)

    set_target_properties(studiocast-resize-backend-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_executable(studiocast-maxine-resident-fake-bench
            ${PROJECT_SOURCE_DIR}/src/tools/studiocast_maxine_resident_fake_bench_main.cpp
            ${PROJECT_SOURCE_DIR}/tests/support/maxine/fake_resident_frame_executor.cpp
    )

    target_link_libraries(studiocast-maxine-resident-fake-bench PRIVATE
            studiocast_maxine_backend
    )

    target_include_directories(studiocast-maxine-resident-fake-bench PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-maxine-resident-fake-bench)
    studiocast_enable_sanitizers(studiocast-maxine-resident-fake-bench)

    set_target_properties(studiocast-maxine-resident-fake-bench PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
endif()


# -----------------------------
# Daemon control CLI (studiocastctl)
# -----------------------------
add_executable(studiocastctl
        ${PROJECT_SOURCE_DIR}/src/tools/studiocastctl_main.cpp
)

target_link_libraries(studiocastctl PRIVATE
        studiocast_contracts
        studiocast_ipc
        studiocast_util
        studiocast_video_io)

target_include_directories(studiocastctl PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
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
