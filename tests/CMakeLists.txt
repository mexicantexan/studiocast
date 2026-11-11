if (BUILD_TESTING AND PULSE_SIMPLE_FOUND)
    add_executable(studiocast-audio-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/audio/audio_service_tests.cpp
    )

    target_link_libraries(studiocast-audio-tests PRIVATE
            studiocast_audio_runtime)

    target_include_directories(studiocast-audio-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-audio-tests integration audio)
elseif (BUILD_TESTING)
    message(STATUS "Skipping studiocast-audio-tests because libpulse-simple was not found.")
endif()
if (BUILD_TESTING)
    add_executable(studiocast-hal-contract-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/hal_contract_tests.cpp
    )

    target_link_libraries(studiocast-hal-contract-tests PRIVATE
            studiocast_hal_contracts)

    target_include_directories(studiocast-hal-contract-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(studiocast-hal-contract-tests)
    studiocast_enable_sanitizers(studiocast-hal-contract-tests)

    set_target_properties(studiocast-hal-contract-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-hal-contract-tests
            COMMAND studiocast-hal-contract-tests)
    studiocast_set_test_labels(studiocast-hal-contract-tests
            unit core video gpu)

    add_executable(studiocast-hal-resolver-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/hal_resolver_tests.cpp
    )

    target_link_libraries(studiocast-hal-resolver-tests PRIVATE
            studiocast_hal_runtime)

    target_include_directories(studiocast-hal-resolver-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(studiocast-hal-resolver-tests)
    studiocast_enable_sanitizers(studiocast-hal-resolver-tests)

    set_target_properties(studiocast-hal-resolver-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-hal-resolver-tests
            COMMAND studiocast-hal-resolver-tests)
    studiocast_set_test_labels(studiocast-hal-resolver-tests
            unit core video gpu)

    add_executable(studiocast-hardware-execution-runtime-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/hardware_execution_runtime_tests.cpp
    )

    target_link_libraries(studiocast-hardware-execution-runtime-tests PRIVATE
            studiocast_video_execution_runtime)

    target_include_directories(studiocast-hardware-execution-runtime-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(studiocast-hardware-execution-runtime-tests)
    studiocast_enable_sanitizers(studiocast-hardware-execution-runtime-tests)

    set_target_properties(studiocast-hardware-execution-runtime-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-hardware-execution-runtime-tests
            COMMAND studiocast-hardware-execution-runtime-tests)
    studiocast_set_test_labels(studiocast-hardware-execution-runtime-tests
            unit core video gpu)

    add_executable(studiocast-video-execution-coordinator-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/video_execution_coordinator_tests.cpp
    )

    target_link_libraries(studiocast-video-execution-coordinator-tests PRIVATE
            studiocast_video_execution_runtime)

    target_include_directories(studiocast-video-execution-coordinator-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-video-execution-coordinator-tests)
    studiocast_enable_sanitizers(
            studiocast-video-execution-coordinator-tests)

    set_target_properties(studiocast-video-execution-coordinator-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-video-execution-coordinator-tests
            COMMAND studiocast-video-execution-coordinator-tests)
    studiocast_set_test_labels(studiocast-video-execution-coordinator-tests
            unit core video gpu)

    add_executable(studiocast-production-video-setup-lease-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/production_video_setup_lease_tests.cpp
    )

    target_link_libraries(
            studiocast-production-video-setup-lease-tests PRIVATE
            studiocast_video_execution_runtime)

    target_include_directories(
            studiocast-production-video-setup-lease-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-production-video-setup-lease-tests)
    studiocast_enable_sanitizers(
            studiocast-production-video-setup-lease-tests)

    set_target_properties(
            studiocast-production-video-setup-lease-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-production-video-setup-lease-tests
            COMMAND studiocast-production-video-setup-lease-tests)
    studiocast_set_test_labels(
            studiocast-production-video-setup-lease-tests
            unit core video gpu)

    add_executable(studiocast-production-video-backend-manager-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/production_video_backend_manager_tests.cpp
    )

    target_link_libraries(
            studiocast-production-video-backend-manager-tests PRIVATE
            studiocast_video_backend_composition)

    target_include_directories(
            studiocast-production-video-backend-manager-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-production-video-backend-manager-tests)
    studiocast_enable_sanitizers(
            studiocast-production-video-backend-manager-tests)

    set_target_properties(
            studiocast-production-video-backend-manager-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-production-video-backend-manager-tests
            COMMAND studiocast-production-video-backend-manager-tests)
    studiocast_set_test_labels(
            studiocast-production-video-backend-manager-tests
            unit core video gpu)

    add_executable(studiocast-production-video-execution-composition-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/production_video_execution_composition_tests.cpp
    )

    target_link_libraries(
            studiocast-production-video-execution-composition-tests PRIVATE
            studiocast_video_backend_composition)

    target_include_directories(
            studiocast-production-video-execution-composition-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-production-video-execution-composition-tests)
    studiocast_enable_sanitizers(
            studiocast-production-video-execution-composition-tests)

    set_target_properties(
            studiocast-production-video-execution-composition-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-production-video-execution-composition-tests
            COMMAND studiocast-production-video-execution-composition-tests)
    studiocast_set_test_labels(
            studiocast-production-video-execution-composition-tests
            integration core video gpu)

    add_executable(studiocast-production-vulkan-pipeline-migration-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/production_vulkan_pipeline_migration_tests.cpp
    )

    target_compile_definitions(
            studiocast-production-vulkan-pipeline-migration-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    studiocast_set_project_warnings(
            studiocast-production-vulkan-pipeline-migration-tests)
    studiocast_enable_sanitizers(
            studiocast-production-vulkan-pipeline-migration-tests)

    set_target_properties(
            studiocast-production-vulkan-pipeline-migration-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-production-vulkan-pipeline-migration-tests
            COMMAND studiocast-production-vulkan-pipeline-migration-tests)
    studiocast_set_test_labels(
            studiocast-production-vulkan-pipeline-migration-tests
            integration core video gpu vulkan)

    add_executable(studiocast-maxine-video-execution-factory-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_video_execution_factory_tests.cpp
    )

    target_link_libraries(studiocast-maxine-video-execution-factory-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-maxine-video-execution-factory-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-maxine-video-execution-factory-tests)
    studiocast_enable_sanitizers(
            studiocast-maxine-video-execution-factory-tests)

    set_target_properties(studiocast-maxine-video-execution-factory-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-maxine-video-execution-factory-tests
            COMMAND studiocast-maxine-video-execution-factory-tests)
    studiocast_set_test_labels(studiocast-maxine-video-execution-factory-tests
            unit core video gpu maxine)

    add_executable(studiocast-open-cuda-video-execution-factory-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_video_execution_factory_tests.cpp
    )

    target_link_libraries(studiocast-open-cuda-video-execution-factory-tests PRIVATE
            studiocast_open_video_backend
            studiocast_video_execution_runtime)

    target_include_directories(studiocast-open-cuda-video-execution-factory-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-open-cuda-video-execution-factory-tests)
    studiocast_enable_sanitizers(
            studiocast-open-cuda-video-execution-factory-tests)

    set_target_properties(
            studiocast-open-cuda-video-execution-factory-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-cuda-video-execution-factory-tests
            COMMAND studiocast-open-cuda-video-execution-factory-tests)
    studiocast_set_test_labels(
            studiocast-open-cuda-video-execution-factory-tests
            unit core video gpu model)

    add_executable(studiocast-vulkan-video-execution-factory-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/video_execution_factory_tests.cpp
    )

    target_link_libraries(studiocast-vulkan-video-execution-factory-tests PRIVATE
            studiocast_video_execution_runtime
            studiocast_vulkan_backend)

    target_include_directories(studiocast-vulkan-video-execution-factory-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(
            studiocast-vulkan-video-execution-factory-tests)
    studiocast_enable_sanitizers(
            studiocast-vulkan-video-execution-factory-tests)

    set_target_properties(
            studiocast-vulkan-video-execution-factory-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-vulkan-video-execution-factory-tests
            COMMAND studiocast-vulkan-video-execution-factory-tests)
    studiocast_set_test_labels(
            studiocast-vulkan-video-execution-factory-tests
            unit core video gpu vulkan)

    add_executable(studiocast-prepared-hardware-execution-owner-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/prepared_hardware_execution_owner_tests.cpp
    )

    target_link_libraries(studiocast-prepared-hardware-execution-owner-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-prepared-hardware-execution-owner-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-prepared-hardware-execution-owner-tests)
    studiocast_enable_sanitizers(studiocast-prepared-hardware-execution-owner-tests)

    set_target_properties(studiocast-prepared-hardware-execution-owner-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-prepared-hardware-execution-owner-tests
            COMMAND studiocast-prepared-hardware-execution-owner-tests)
    studiocast_set_test_labels(studiocast-prepared-hardware-execution-owner-tests
            integration core video gpu vulkan)

    add_executable(studiocast-prepared-maxine-hardware-execution-owner-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/prepared_maxine_hardware_execution_owner_tests.cpp
    )

    target_link_libraries(studiocast-prepared-maxine-hardware-execution-owner-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-prepared-maxine-hardware-execution-owner-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-prepared-maxine-hardware-execution-owner-tests)
    studiocast_enable_sanitizers(studiocast-prepared-maxine-hardware-execution-owner-tests)

    set_target_properties(studiocast-prepared-maxine-hardware-execution-owner-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-prepared-maxine-hardware-execution-owner-tests
            COMMAND studiocast-prepared-maxine-hardware-execution-owner-tests)
    studiocast_set_test_labels(studiocast-prepared-maxine-hardware-execution-owner-tests
            integration core video gpu maxine)

    add_executable(studiocast-prepared-open-cuda-hardware-execution-owner-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/prepared_open_cuda_hardware_execution_owner_tests.cpp
    )

    target_link_libraries(studiocast-prepared-open-cuda-hardware-execution-owner-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-prepared-open-cuda-hardware-execution-owner-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-prepared-open-cuda-hardware-execution-owner-tests)
    studiocast_enable_sanitizers(
            studiocast-prepared-open-cuda-hardware-execution-owner-tests)

    set_target_properties(
            studiocast-prepared-open-cuda-hardware-execution-owner-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-prepared-open-cuda-hardware-execution-owner-tests
            COMMAND studiocast-prepared-open-cuda-hardware-execution-owner-tests)
    studiocast_set_test_labels(
            studiocast-prepared-open-cuda-hardware-execution-owner-tests
            integration core video gpu model)

    add_executable(studiocast-prepared-open-cuda-pipeline-migration-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/prepared_open_cuda_pipeline_migration_tests.cpp
    )

    target_link_libraries(studiocast-prepared-open-cuda-pipeline-migration-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-prepared-open-cuda-pipeline-migration-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(
            studiocast-prepared-open-cuda-pipeline-migration-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    studiocast_set_project_warnings(
            studiocast-prepared-open-cuda-pipeline-migration-tests)
    studiocast_enable_sanitizers(
            studiocast-prepared-open-cuda-pipeline-migration-tests)

    set_target_properties(
            studiocast-prepared-open-cuda-pipeline-migration-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-prepared-open-cuda-pipeline-migration-tests
            COMMAND studiocast-prepared-open-cuda-pipeline-migration-tests)
    studiocast_set_test_labels(
            studiocast-prepared-open-cuda-pipeline-migration-tests
            integration core video gpu model)

    add_executable(studiocast-production-backend-pipeline-lease-cutover-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/production_backend_pipeline_lease_cutover_tests.cpp
    )

    target_compile_definitions(
            studiocast-production-backend-pipeline-lease-cutover-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    studiocast_set_project_warnings(
            studiocast-production-backend-pipeline-lease-cutover-tests)
    studiocast_enable_sanitizers(
            studiocast-production-backend-pipeline-lease-cutover-tests)

    set_target_properties(
            studiocast-production-backend-pipeline-lease-cutover-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-production-backend-pipeline-lease-cutover-tests
            COMMAND studiocast-production-backend-pipeline-lease-cutover-tests)
    studiocast_set_test_labels(
            studiocast-production-backend-pipeline-lease-cutover-tests
            integration core video gpu model maxine)

    add_executable(studiocast-video-runtime-status-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/video_runtime_status_tests.cpp
    )

    target_link_libraries(studiocast-video-runtime-status-tests PRIVATE
            studiocast_video_execution_runtime)

    target_include_directories(studiocast-video-runtime-status-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    studiocast_set_project_warnings(studiocast-video-runtime-status-tests)
    studiocast_enable_sanitizers(studiocast-video-runtime-status-tests)

    set_target_properties(studiocast-video-runtime-status-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-video-runtime-status-tests
            COMMAND studiocast-video-runtime-status-tests)
    studiocast_set_test_labels(studiocast-video-runtime-status-tests
            unit core video gpu)

    add_executable(studiocast-daemon-status-json-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/daemon/status_json_tests.cpp
    )

    target_link_libraries(studiocast-daemon-status-json-tests PRIVATE
            studiocast_daemon_status
            studiocast_util)

    target_include_directories(studiocast-daemon-status-json-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    target_compile_definitions(studiocast-daemon-status-json-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

    studiocast_set_project_warnings(studiocast-daemon-status-json-tests)
    studiocast_enable_sanitizers(studiocast-daemon-status-json-tests)

    set_target_properties(studiocast-daemon-status-json-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-daemon-status-json-tests
            COMMAND studiocast-daemon-status-json-tests)
    studiocast_set_test_labels(studiocast-daemon-status-json-tests
            unit core video gpu)

    add_library(studiocast-maxine-effect-fake-sdk SHARED
            ${PROJECT_SOURCE_DIR}/tests/support/maxine/maxine_effect_runtime_fake_sdk.cpp
    )
    target_include_directories(studiocast-maxine-effect-fake-sdk PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support/maxine"
    )
    set_target_properties(studiocast-maxine-effect-fake-sdk PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
    studiocast_set_project_warnings(studiocast-maxine-effect-fake-sdk)

    add_library(studiocast-maxine-effect-fake-cudart SHARED
            ${PROJECT_SOURCE_DIR}/tests/support/maxine/maxine_effect_runtime_fake_cudart.cpp
    )
    set_target_properties(studiocast-maxine-effect-fake-cudart PROPERTIES
            OUTPUT_NAME cudart
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
    studiocast_set_project_warnings(studiocast-maxine-effect-fake-cudart)

    add_executable(studiocast-maxine-effect-binding-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_effect_binding_tests.cpp
    )
    target_link_libraries(studiocast-maxine-effect-binding-tests PRIVATE
            studiocast_maxine_backend
            studiocast-maxine-effect-fake-sdk
    )
    target_include_directories(studiocast-maxine-effect-binding-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support/maxine"
            "${STUDIOCAST_GENERATED_DIR}"
    )
    target_compile_definitions(studiocast-maxine-effect-binding-tests PRIVATE
            STUDIOCAST_FAKE_MAXINE_SDK_PATH="$<TARGET_FILE:studiocast-maxine-effect-fake-sdk>"
    )
    studiocast_set_project_warnings(studiocast-maxine-effect-binding-tests)
    studiocast_enable_sanitizers(studiocast-maxine-effect-binding-tests)
    set_target_properties(studiocast-maxine-effect-binding-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
    add_dependencies(studiocast-maxine-effect-binding-tests
            studiocast-maxine-effect-fake-cudart)
    add_test(NAME studiocast-maxine-effect-binding-tests
            COMMAND studiocast-maxine-effect-binding-tests)
    set_tests_properties(studiocast-maxine-effect-binding-tests PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:studiocast-maxine-effect-fake-cudart>"
    )
    studiocast_set_test_labels(studiocast-maxine-effect-binding-tests
            integration gpu maxine video)

    add_executable(studiocast-maxine-production-resident-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_production_resident_executor_tests.cpp
    )
    target_link_libraries(studiocast-maxine-production-resident-tests PRIVATE
            studiocast_maxine_backend
            studiocast-maxine-effect-fake-sdk
    )
    target_include_directories(studiocast-maxine-production-resident-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support/maxine"
            "${STUDIOCAST_GENERATED_DIR}"
    )
    target_compile_definitions(studiocast-maxine-production-resident-tests PRIVATE
            STUDIOCAST_FAKE_MAXINE_SDK_PATH="$<TARGET_FILE:studiocast-maxine-effect-fake-sdk>"
    )
    studiocast_set_project_warnings(studiocast-maxine-production-resident-tests)
    studiocast_enable_sanitizers(studiocast-maxine-production-resident-tests)
    set_target_properties(studiocast-maxine-production-resident-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
    add_dependencies(studiocast-maxine-production-resident-tests
            studiocast-maxine-effect-fake-cudart)
    add_test(NAME studiocast-maxine-production-resident-tests
            COMMAND studiocast-maxine-production-resident-tests)
    set_tests_properties(studiocast-maxine-production-resident-tests PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:studiocast-maxine-effect-fake-cudart>"
    )
    studiocast_set_test_labels(studiocast-maxine-production-resident-tests
            integration gpu maxine video)

    add_executable(studiocast-maxine-production-hal-owner-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_production_hal_owner_tests.cpp
    )
    target_link_libraries(studiocast-maxine-production-hal-owner-tests PRIVATE
            studiocast_maxine_backend
            studiocast_video_execution_runtime
            studiocast-maxine-effect-fake-sdk
    )
    target_include_directories(studiocast-maxine-production-hal-owner-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support/maxine"
            "${STUDIOCAST_GENERATED_DIR}"
    )
    target_compile_definitions(studiocast-maxine-production-hal-owner-tests PRIVATE
            STUDIOCAST_FAKE_MAXINE_SDK_PATH="$<TARGET_FILE:studiocast-maxine-effect-fake-sdk>"
    )
    studiocast_set_project_warnings(studiocast-maxine-production-hal-owner-tests)
    studiocast_enable_sanitizers(studiocast-maxine-production-hal-owner-tests)
    set_target_properties(studiocast-maxine-production-hal-owner-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )
    add_dependencies(studiocast-maxine-production-hal-owner-tests
            studiocast-maxine-effect-fake-cudart)
    add_test(NAME studiocast-maxine-production-hal-owner-tests
            COMMAND studiocast-maxine-production-hal-owner-tests)
    set_tests_properties(studiocast-maxine-production-hal-owner-tests PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:studiocast-maxine-effect-fake-cudart>"
    )
    studiocast_set_test_labels(studiocast-maxine-production-hal-owner-tests
            integration gpu maxine video)

    if (STUDIOCAST_BUILD_BENCHMARKS)
        add_executable(studiocast-maxine-production-resident-fake-bench
                ${PROJECT_SOURCE_DIR}/src/tools/studiocast_maxine_production_resident_fake_bench_main.cpp
        )
        target_link_libraries(studiocast-maxine-production-resident-fake-bench PRIVATE
                studiocast_maxine_backend
                studiocast-maxine-effect-fake-sdk
        )
        target_include_directories(studiocast-maxine-production-resident-fake-bench PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
                "${PROJECT_SOURCE_DIR}/tests/support/maxine"
                "${STUDIOCAST_GENERATED_DIR}"
        )
        target_compile_definitions(studiocast-maxine-production-resident-fake-bench PRIVATE
                STUDIOCAST_FAKE_MAXINE_SDK_PATH="$<TARGET_FILE:studiocast-maxine-effect-fake-sdk>"
        )
        studiocast_set_project_warnings(studiocast-maxine-production-resident-fake-bench)
        studiocast_enable_sanitizers(studiocast-maxine-production-resident-fake-bench)
        set_target_properties(studiocast-maxine-production-resident-fake-bench PROPERTIES
                AUTOMOC OFF
                AUTOUIC OFF
                AUTORCC OFF
        )
        add_dependencies(studiocast-maxine-production-resident-fake-bench
                studiocast-maxine-effect-fake-cudart)
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME studiocast-embedded-ptx-freshness
                COMMAND "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/gpu/cuda/validate_embedded_ptx.py"
                        --source-root "${PROJECT_SOURCE_DIR}")
        set_tests_properties(studiocast-embedded-ptx-freshness PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
        studiocast_set_test_labels(studiocast-embedded-ptx-freshness unit gpu maxine)
        add_test(NAME studiocast-embedded-vulkan-spv-freshness
                COMMAND "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/validate_embedded_vulkan_spv.py"
                        --source-root "${PROJECT_SOURCE_DIR}")
        set_tests_properties(studiocast-embedded-vulkan-spv-freshness PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
        studiocast_set_test_labels(studiocast-embedded-vulkan-spv-freshness unit gpu vulkan)
    else()
        message(STATUS "Skipping embedded shader freshness tests because Python3 was not found.")
    endif()

    add_executable(studiocast-maxine-paths-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_paths_tests.cpp
    )

    target_link_libraries(studiocast-maxine-paths-tests PRIVATE
            studiocast_maxine_backend)

    target_include_directories(studiocast-maxine-paths-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-maxine-paths-tests
            integration gpu maxine audio video)

    add_executable(studiocast-nvcv-validation-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/nvcv_validation_tests.cpp
    )

    target_link_libraries(studiocast-nvcv-validation-tests PRIVATE
            studiocast_maxine_backend
    )

    target_include_directories(studiocast-nvcv-validation-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-nvcv-validation-tests unit gpu maxine video)

    add_executable(studiocast-maxine-resident-frame-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_resident_frame_section_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/support/maxine/fake_resident_frame_executor.cpp
    )

    target_link_libraries(studiocast-maxine-resident-frame-tests PRIVATE
            studiocast_maxine_backend
    )

    target_include_directories(studiocast-maxine-resident-frame-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-maxine-resident-frame-tests)
    studiocast_enable_sanitizers(studiocast-maxine-resident-frame-tests)

    set_target_properties(studiocast-maxine-resident-frame-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-maxine-resident-frame-tests
            COMMAND studiocast-maxine-resident-frame-tests)
    studiocast_set_test_labels(studiocast-maxine-resident-frame-tests
            unit gpu maxine video)

    add_executable(studiocast-maxine-hal-resident-adapter-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/maxine/maxine_hal_resident_adapter_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/support/maxine/fake_resident_frame_executor.cpp
    )

    target_link_libraries(studiocast-maxine-hal-resident-adapter-tests PRIVATE
            studiocast_maxine_backend
    )

    target_include_directories(studiocast-maxine-hal-resident-adapter-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/tests/support"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-maxine-hal-resident-adapter-tests)
    studiocast_enable_sanitizers(studiocast-maxine-hal-resident-adapter-tests)

    set_target_properties(studiocast-maxine-hal-resident-adapter-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-maxine-hal-resident-adapter-tests
            COMMAND studiocast-maxine-hal-resident-adapter-tests)
    studiocast_set_test_labels(studiocast-maxine-hal-resident-adapter-tests
            unit gpu maxine video)

    add_executable(studiocast-model-registry-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/model/model_pack_registry_tests.cpp
    )

    target_link_libraries(studiocast-model-registry-tests PRIVATE
            studiocast_open_audio_backend
            studiocast_open_video_backend)

    target_include_directories(studiocast-model-registry-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
    studiocast_set_test_labels(studiocast-model-registry-tests
            unit model audio video gpu)

    add_executable(studiocast-open-video-strict-provider-artifact-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_video_strict_provider_artifact_tests.cpp
    )

    target_link_libraries(studiocast-open-video-strict-provider-artifact-tests PRIVATE
            studiocast_onnx_support
            studiocast_model_support
    )

    target_include_directories(studiocast-open-video-strict-provider-artifact-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-open-video-strict-provider-artifact-tests)
    studiocast_enable_sanitizers(
            studiocast-open-video-strict-provider-artifact-tests)

    set_target_properties(
            studiocast-open-video-strict-provider-artifact-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-video-strict-provider-artifact-tests
            COMMAND studiocast-open-video-strict-provider-artifact-tests)
    studiocast_set_test_labels(
            studiocast-open-video-strict-provider-artifact-tests
            unit gpu video model)

    add_executable(studiocast-open-video-typed-run-contract-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_video_typed_run_contract_tests.cpp
    )

    target_link_libraries(studiocast-open-video-typed-run-contract-tests PRIVATE
            studiocast_open_video_backend
    )

    target_include_directories(studiocast-open-video-typed-run-contract-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-open-video-typed-run-contract-tests)
    studiocast_enable_sanitizers(studiocast-open-video-typed-run-contract-tests)

    set_target_properties(studiocast-open-video-typed-run-contract-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-video-typed-run-contract-tests
            COMMAND studiocast-open-video-typed-run-contract-tests)
    studiocast_set_test_labels(studiocast-open-video-typed-run-contract-tests
            unit video gpu model)

    add_executable(studiocast-open-audio-prepared-binding-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/audio/open_audio_prepared_binding_tests.cpp
    )

    target_link_libraries(studiocast-open-audio-prepared-binding-tests PRIVATE
            studiocast_open_audio_backend
    )

    target_include_directories(studiocast-open-audio-prepared-binding-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-open-audio-prepared-binding-tests)
    studiocast_enable_sanitizers(studiocast-open-audio-prepared-binding-tests)

    set_target_properties(studiocast-open-audio-prepared-binding-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-audio-prepared-binding-tests
            COMMAND studiocast-open-audio-prepared-binding-tests)
    studiocast_set_test_labels(studiocast-open-audio-prepared-binding-tests
            unit audio model)

    add_executable(studiocast-cuda-driver-api-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/cuda_driver_api_tests.cpp
    )

    target_link_libraries(studiocast-cuda-driver-api-tests PRIVATE
            studiocast_cuda_support
    )

    target_include_directories(studiocast-cuda-driver-api-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-cuda-driver-api-tests unit gpu)

    add_executable(studiocast-cuda-prepared-transfer-completion-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/cuda_prepared_transfer_completion_tests.cpp
    )

    target_link_libraries(studiocast-cuda-prepared-transfer-completion-tests PRIVATE
            studiocast_cuda_support
    )

    target_include_directories(studiocast-cuda-prepared-transfer-completion-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-cuda-prepared-transfer-completion-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-transfer-completion-tests)

    set_target_properties(studiocast-cuda-prepared-transfer-completion-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-cuda-prepared-transfer-completion-tests
            COMMAND studiocast-cuda-prepared-transfer-completion-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-transfer-completion-tests
            unit gpu)

    add_executable(studiocast-cuda-prepared-preprocess-kernel-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_preprocess_kernel_bundle_tests.cpp
    )

    target_link_libraries(studiocast-cuda-prepared-preprocess-kernel-tests PRIVATE
            studiocast_cuda_support
    )

    target_include_directories(studiocast-cuda-prepared-preprocess-kernel-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-cuda-prepared-preprocess-kernel-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-preprocess-kernel-tests)

    set_target_properties(studiocast-cuda-prepared-preprocess-kernel-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-cuda-prepared-preprocess-kernel-tests
            COMMAND studiocast-cuda-prepared-preprocess-kernel-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-preprocess-kernel-tests
            unit gpu)

    add_executable(studiocast-cuda-prepared-vignette-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_vignette_tests.cpp)
    target_link_libraries(studiocast-cuda-prepared-vignette-tests PRIVATE
            studiocast_cuda_support)
    target_include_directories(studiocast-cuda-prepared-vignette-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}")
    studiocast_set_project_warnings(studiocast-cuda-prepared-vignette-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-vignette-tests)
    set_target_properties(studiocast-cuda-prepared-vignette-tests PROPERTIES
            AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME studiocast-cuda-prepared-vignette-tests
            COMMAND studiocast-cuda-prepared-vignette-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-vignette-tests unit gpu)

    add_executable(studiocast-cuda-prepared-vb-kernel-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_open_cuda_vb_kernel_bundle_tests.cpp
    )

    target_link_libraries(studiocast-cuda-prepared-vb-kernel-tests PRIVATE
            studiocast_cuda_support
    )

    target_include_directories(studiocast-cuda-prepared-vb-kernel-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-cuda-prepared-vb-kernel-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-vb-kernel-tests)

    set_target_properties(studiocast-cuda-prepared-vb-kernel-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-cuda-prepared-vb-kernel-tests
            COMMAND studiocast-cuda-prepared-vb-kernel-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-vb-kernel-tests unit gpu)

    add_executable(studiocast-ipc-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/ipc/ipc_client_tests.cpp
    )

    target_link_libraries(studiocast-ipc-tests PRIVATE studiocast_ipc)

    target_include_directories(studiocast-ipc-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-ipc-tests integration ipc core)

    add_executable(studiocast-gui-status-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/gui/gui_status_snapshot_tests.cpp
            ${PROJECT_SOURCE_DIR}/src/gui/text_edit_utils.h
            ${PROJECT_SOURCE_DIR}/src/gui/text_edit_utils.cpp
            ${PROJECT_SOURCE_DIR}/src/gui/pages/engines_models_page.h
            ${PROJECT_SOURCE_DIR}/src/gui/pages/engines_models_page.cpp
            ${PROJECT_SOURCE_DIR}/src/gui/pages/video_page.h
            ${PROJECT_SOURCE_DIR}/src/gui/pages/video_page.cpp
            ${PROJECT_SOURCE_DIR}/src/gui/status/daemon_status_snapshot.h
            ${PROJECT_SOURCE_DIR}/src/gui/status/daemon_status_snapshot.cpp
            ${PROJECT_SOURCE_DIR}/src/gui/status/pending_daemon_write_guard.h
            ${PROJECT_SOURCE_DIR}/src/gui/status/status_poller.h
            ${PROJECT_SOURCE_DIR}/src/gui/status/status_poller.cpp
    )

    target_link_libraries(studiocast-gui-status-tests PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
            studiocast_audio_runtime
            studiocast_contracts
            studiocast_ipc
            studiocast_video_io
            studiocast_video_support
    )

    target_include_directories(studiocast-gui-status-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-gui-status-tests
            integration gui ipc audio video model gpu vulkan)

    add_executable(studiocast-effect-schema-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/effect_schema_tests.cpp
    )

    target_link_libraries(studiocast-effect-schema-tests PRIVATE studiocast_contracts)

    target_include_directories(studiocast-effect-schema-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-effect-schema-tests unit core audio video)

    add_executable(studiocast-daemon-status-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/core/daemon_status_tests.cpp
    )

    target_link_libraries(studiocast-daemon-status-tests PRIVATE
            studiocast_daemon_status
            studiocast_ipc
            studiocast_runtime_config)

    target_include_directories(studiocast-daemon-status-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-daemon-status-tests
            integration core audio video gpu vulkan)

    add_executable(studiocast-exec-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/core/exec_tests.cpp
    )

    target_link_libraries(studiocast-exec-tests PRIVATE studiocast_util)

    target_include_directories(studiocast-exec-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-exec-tests unit core)

    add_executable(studiocast-installer-backend-tests
            ${PROJECT_SOURCE_DIR}/tests/installer/backend/installer_backend_tests.cpp
    )

    target_link_libraries(studiocast-installer-backend-tests PRIVATE
            studiocast_util
    )

    target_include_directories(studiocast-installer-backend-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-installer-backend-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            STUDIOCAST_BINARY_DIR="${PROJECT_BINARY_DIR}"
            STUDIOCAST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
    )

    add_dependencies(studiocast-installer-backend-tests studiocastd studiocastctl)

    studiocast_set_project_warnings(studiocast-installer-backend-tests)
    studiocast_enable_sanitizers(studiocast-installer-backend-tests)

    set_target_properties(studiocast-installer-backend-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-installer-backend-tests
            COMMAND studiocast-installer-backend-tests)
    studiocast_set_test_labels(studiocast-installer-backend-tests
            system installer core)

    if (Python3_Interpreter_FOUND)
        add_test(NAME studiocast-documentation-validation-tests
                COMMAND "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/docs/validate_documentation.py"
                        --source-root "${PROJECT_SOURCE_DIR}")
        set_tests_properties(studiocast-documentation-validation-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-documentation-validation-tests
                unit core)

        add_test(NAME studiocast-hardware-ci-workflow-tests
                COMMAND "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/gpu/hardware_ci_workflow_tests.py")
        studiocast_set_test_labels(studiocast-hardware-ci-workflow-tests
                unit gpu video)

        add_test(NAME studiocast-fastdvdnet-export-tests
                COMMAND "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/models/fastdvdnet_export_tests.py")
        studiocast_set_test_labels(studiocast-fastdvdnet-export-tests
                unit installer model)

        add_test(NAME studiocast-installer-backend-core-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/backend/installer_backend_core_tests.py")
        set_tests_properties(studiocast-installer-backend-core-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-installer-backend-core-tests
                system installer release model gpu vulkan)
        add_test(NAME studiocast-model-download-transaction-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/models/model_download_transaction_tests.py")
        set_tests_properties(studiocast-model-download-transaction-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-model-download-transaction-tests
                integration installer model)
        add_test(NAME studiocast-installer-packaging-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/packaging/installer_packaging_tests.py")
        set_tests_properties(studiocast-installer-packaging-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-installer-packaging-tests
                system installer release model)

        add_test(NAME studiocast-release-channel-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/release/release_channel_tests.py")
        set_tests_properties(studiocast-release-channel-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-release-channel-tests unit release)

        add_test(NAME studiocast-release-packaging-security-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/release/release_packaging_security_tests.py")
        set_tests_properties(studiocast-release-packaging-security-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-release-packaging-security-tests
                integration installer release)

        add_test(NAME studiocast-privileged-helper-tests
                COMMAND "${CMAKE_COMMAND}" -E env
                        "STUDIOCAST_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "${Python3_EXECUTABLE}"
                        "${PROJECT_SOURCE_DIR}/tests/installer/privileged/privileged_helper_tests.py")
        set_tests_properties(studiocast-privileged-helper-tests PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        studiocast_set_test_labels(studiocast-privileged-helper-tests
                integration installer ipc)
    endif()

    add_executable(studiocast-installer-wizard-tests
            ${PROJECT_SOURCE_DIR}/tests/installer/gui/installer_wizard_tests.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard.h
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_internal.h
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_support.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_intro_pages.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_setup_pages.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_review_pages.cpp
            ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard_progress.cpp
    )

    target_link_libraries(studiocast-installer-wizard-tests PRIVATE
            Qt${QT_VERSION_MAJOR}::Widgets
    )

    target_include_directories(studiocast-installer-wizard-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/installer/gui"
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-installer-wizard-tests PRIVATE
            STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH="${PROJECT_SOURCE_DIR}/installer/backend/studiocast-installer-backend"
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
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
    studiocast_set_test_labels(studiocast-installer-wizard-tests
            integration installer gui release model)

    add_executable(studiocast-cmake-open-backend-cache-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/core/cmake_open_backend_cache_tests.cpp
    )

    target_link_libraries(studiocast-cmake-open-backend-cache-tests PRIVATE
            studiocast_util
    )

    target_include_directories(studiocast-cmake-open-backend-cache-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    target_compile_definitions(studiocast-cmake-open-backend-cache-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
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
    studiocast_set_test_labels(studiocast-cmake-open-backend-cache-tests
            system core gpu)

    add_executable(studiocast-video-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/video_service_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/camera_effects_update_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/latest_frame_wins_worker_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_video_denoise_contract_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/frame_artifact_cache_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/cuda_kernel_cache_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/open_vulkan_matting_setup_policy_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/replace_background_cache_policy_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/v4l2_capture_negotiation_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/video_effect_cpu_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/rgb_yuyv_convert_tests.cpp
            ${PROJECT_SOURCE_DIR}/tests/unit/video/yuyv_passthrough_tests.cpp
    )

    target_link_libraries(studiocast-video-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-video-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    # The video runtime deliberately does not propagate optional-backend feature
    # definitions. Matrix-aware contract tests still need the configured build
    # state to select the matching fail-closed expectations.
    target_compile_definitions(studiocast-video-tests PRIVATE
            STUDIOCAST_ENABLE_OPEN_VULKAN=$<BOOL:${STUDIOCAST_ENABLE_OPEN_VULKAN}>
    )

    studiocast_set_project_warnings(studiocast-video-tests)
    studiocast_enable_sanitizers(studiocast-video-tests)

    set_target_properties(studiocast-video-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-video-tests COMMAND studiocast-video-tests)
    studiocast_set_test_labels(studiocast-video-tests
            integration video gpu vulkan maxine)

    add_executable(studiocast-v4l2-writer-transport-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/v4l2_writer_transport_tests.cpp
    )

    target_link_libraries(studiocast-v4l2-writer-transport-tests PRIVATE
            studiocast_video_io
    )

    target_include_directories(studiocast-v4l2-writer-transport-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-v4l2-writer-transport-tests)
    studiocast_enable_sanitizers(studiocast-v4l2-writer-transport-tests)

    set_target_properties(studiocast-v4l2-writer-transport-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-v4l2-writer-transport-tests
            COMMAND studiocast-v4l2-writer-transport-tests)
    studiocast_set_test_labels(studiocast-v4l2-writer-transport-tests
            integration video)

    add_executable(studiocast-open-video-scratch-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_video_scratch_reuse_tests.cpp
    )

    target_link_libraries(studiocast-open-video-scratch-tests PRIVATE
            studiocast_open_video_backend)

    target_include_directories(studiocast-open-video-scratch-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-open-video-scratch-tests)
    studiocast_enable_sanitizers(studiocast-open-video-scratch-tests)

    set_target_properties(studiocast-open-video-scratch-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-video-scratch-tests
            COMMAND studiocast-open-video-scratch-tests)
    studiocast_set_test_labels(studiocast-open-video-scratch-tests
            unit video gpu model)

    add_executable(studiocast-open-cuda-hal-resident-adapter-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_hal_adapter_tests.cpp
    )

    target_link_libraries(studiocast-open-cuda-hal-resident-adapter-tests PRIVATE
            studiocast_open_video_backend)

    target_include_directories(studiocast-open-cuda-hal-resident-adapter-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-open-cuda-hal-resident-adapter-tests)
    studiocast_enable_sanitizers(studiocast-open-cuda-hal-resident-adapter-tests)

    set_target_properties(studiocast-open-cuda-hal-resident-adapter-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-cuda-hal-resident-adapter-tests
            COMMAND studiocast-open-cuda-hal-resident-adapter-tests)
    studiocast_set_test_labels(studiocast-open-cuda-hal-resident-adapter-tests
            unit gpu video model)

    add_executable(studiocast-open-cuda-prepared-resident-executor-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_prepared_resident_executor_tests.cpp
    )

    target_link_libraries(studiocast-open-cuda-prepared-resident-executor-tests PRIVATE
            studiocast_open_video_backend)

    target_include_directories(studiocast-open-cuda-prepared-resident-executor-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-open-cuda-prepared-resident-executor-tests)
    studiocast_enable_sanitizers(
            studiocast-open-cuda-prepared-resident-executor-tests)

    set_target_properties(
            studiocast-open-cuda-prepared-resident-executor-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-cuda-prepared-resident-executor-tests
            COMMAND studiocast-open-cuda-prepared-resident-executor-tests)
    studiocast_set_test_labels(
            studiocast-open-cuda-prepared-resident-executor-tests
            unit gpu video model)

    add_executable(studiocast-open-cuda-production-hal-owner-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_production_hal_owner_tests.cpp
    )

    target_link_libraries(studiocast-open-cuda-production-hal-owner-tests PRIVATE
            studiocast_open_video_backend
            studiocast_video_execution_runtime)

    target_include_directories(studiocast-open-cuda-production-hal-owner-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-open-cuda-production-hal-owner-tests)
    studiocast_enable_sanitizers(
            studiocast-open-cuda-production-hal-owner-tests)

    set_target_properties(
            studiocast-open-cuda-production-hal-owner-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-open-cuda-production-hal-owner-tests
            COMMAND studiocast-open-cuda-production-hal-owner-tests)
    studiocast_set_test_labels(
            studiocast-open-cuda-production-hal-owner-tests
            unit gpu video model)

    add_executable(
            studiocast-open-cuda-production-backend-setup-source-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_production_backend_setup_source_tests.cpp
    )

    target_link_libraries(
            studiocast-open-cuda-production-backend-setup-source-tests PRIVATE
            studiocast_open_video_backend
            studiocast_video_execution_runtime
            studiocast_video_backend_composition)

    target_include_directories(
            studiocast-open-cuda-production-backend-setup-source-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-open-cuda-production-backend-setup-source-tests)
    studiocast_enable_sanitizers(
            studiocast-open-cuda-production-backend-setup-source-tests)

    set_target_properties(
            studiocast-open-cuda-production-backend-setup-source-tests
            PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(
            NAME studiocast-open-cuda-production-backend-setup-source-tests
            COMMAND studiocast-open-cuda-production-backend-setup-source-tests)
    studiocast_set_test_labels(
            studiocast-open-cuda-production-backend-setup-source-tests
            integration gpu video model)

    add_executable(
            studiocast-open-cuda-production-smoke-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/open_cuda_production_smoke_tests.cpp
    )

    target_link_libraries(
            studiocast-open-cuda-production-smoke-tests PRIVATE
            studiocast_open_video_backend
            studiocast_video_execution_runtime
            studiocast_video_backend_composition)

    target_include_directories(
            studiocast-open-cuda-production-smoke-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-open-cuda-production-smoke-tests)
    studiocast_enable_sanitizers(studiocast-open-cuda-production-smoke-tests)

    set_target_properties(
            studiocast-open-cuda-production-smoke-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(
            NAME studiocast-open-cuda-production-smoke-tests
            COMMAND studiocast-open-cuda-production-smoke-tests)
    set_tests_properties(
            studiocast-open-cuda-production-smoke-tests PROPERTIES
            SKIP_RETURN_CODE 77)
    studiocast_set_test_labels(
            studiocast-open-cuda-production-smoke-tests
            integration gpu video model)

    add_executable(
            studiocast-prepared-open-video-compatibility-executor-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_open_video_compatibility_executor_tests.cpp
    )

    target_link_libraries(
            studiocast-prepared-open-video-compatibility-executor-tests PRIVATE
            studiocast_open_video_backend
            studiocast_video_execution_runtime)

    target_include_directories(
            studiocast-prepared-open-video-compatibility-executor-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(
            studiocast-prepared-open-video-compatibility-executor-tests)
    studiocast_enable_sanitizers(
            studiocast-prepared-open-video-compatibility-executor-tests)

    set_target_properties(
            studiocast-prepared-open-video-compatibility-executor-tests
            PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(
            NAME studiocast-prepared-open-video-compatibility-executor-tests
            COMMAND studiocast-prepared-open-video-compatibility-executor-tests)
    studiocast_set_test_labels(
            studiocast-prepared-open-video-compatibility-executor-tests
            unit gpu video model)

    add_executable(studiocast-video-frame-plan-tests
            ${PROJECT_SOURCE_DIR}/tests/unit/video/video_frame_plan_tests.cpp
    )

    target_link_libraries(studiocast-video-frame-plan-tests PRIVATE
            studiocast_video_runtime)

    target_include_directories(studiocast-video-frame-plan-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-video-frame-plan-tests)
    studiocast_enable_sanitizers(studiocast-video-frame-plan-tests)

    set_target_properties(studiocast-video-frame-plan-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-video-frame-plan-tests
            COMMAND studiocast-video-frame-plan-tests)
    studiocast_set_test_labels(studiocast-video-frame-plan-tests
            unit core video)

    add_executable(studiocast-vulkan-frame-plan-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_frame_execution_plan_tests.cpp
    )

    target_link_libraries(studiocast-vulkan-frame-plan-tests PRIVATE
            studiocast_contracts)

    target_include_directories(studiocast-vulkan-frame-plan-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-vulkan-frame-plan-tests
            unit video gpu vulkan)

    add_executable(studiocast-vulkan-capability-audit-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_capability_audit_tests.cpp
    )

    target_include_directories(studiocast-vulkan-capability-audit-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
    )

    target_compile_definitions(studiocast-vulkan-capability-audit-tests PRIVATE
            STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
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
    studiocast_set_test_labels(studiocast-vulkan-capability-audit-tests
            unit video gpu vulkan installer)

    add_executable(studiocast-vulkan-hal-resident-adapter-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_hal_resident_adapter_tests.cpp
    )

    target_link_libraries(studiocast-vulkan-hal-resident-adapter-tests PRIVATE
            studiocast_vulkan_backend
    )

    target_include_directories(studiocast-vulkan-hal-resident-adapter-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
            "${STUDIOCAST_GENERATED_DIR}"
    )

    studiocast_set_project_warnings(studiocast-vulkan-hal-resident-adapter-tests)
    studiocast_enable_sanitizers(studiocast-vulkan-hal-resident-adapter-tests)

    set_target_properties(studiocast-vulkan-hal-resident-adapter-tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    add_test(NAME studiocast-vulkan-hal-resident-adapter-tests
            COMMAND studiocast-vulkan-hal-resident-adapter-tests)
    studiocast_set_test_labels(studiocast-vulkan-hal-resident-adapter-tests
            unit video gpu vulkan)

    if (STUDIOCAST_ENABLE_OPEN_VULKAN)
        add_executable(studiocast-vulkan-kernel-tests
                ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_kernel_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-kernel-tests PRIVATE
                studiocast_vulkan_backend
        )

        target_include_directories(studiocast-vulkan-kernel-tests PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
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
        studiocast_set_test_labels(studiocast-vulkan-kernel-tests
                integration video gpu vulkan)

        add_executable(studiocast-vulkan-final-stage-integration-tests
                ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_final_resident_stage_integration_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-final-stage-integration-tests PRIVATE
                studiocast_vulkan_backend
        )

        target_include_directories(studiocast-vulkan-final-stage-integration-tests PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
                "${STUDIOCAST_GENERATED_DIR}"
        )

        studiocast_set_project_warnings(studiocast-vulkan-final-stage-integration-tests)
        studiocast_enable_sanitizers(studiocast-vulkan-final-stage-integration-tests)

        set_target_properties(studiocast-vulkan-final-stage-integration-tests PROPERTIES
                AUTOMOC OFF
                AUTOUIC OFF
                AUTORCC OFF
        )

        add_test(NAME studiocast-vulkan-final-stage-integration-tests
                COMMAND studiocast-vulkan-final-stage-integration-tests)
        studiocast_set_test_labels(studiocast-vulkan-final-stage-integration-tests
                integration video gpu vulkan)

        add_executable(studiocast-vulkan-qualified-final-executor-tests
                ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/qualified_final_resident_executor_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-qualified-final-executor-tests PRIVATE
                studiocast_vulkan_backend
        )

        target_include_directories(studiocast-vulkan-qualified-final-executor-tests PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
                "${STUDIOCAST_GENERATED_DIR}"
        )

        studiocast_set_project_warnings(studiocast-vulkan-qualified-final-executor-tests)
        studiocast_enable_sanitizers(studiocast-vulkan-qualified-final-executor-tests)

        set_target_properties(studiocast-vulkan-qualified-final-executor-tests PROPERTIES
                AUTOMOC OFF
                AUTOUIC OFF
                AUTORCC OFF
        )

        add_test(NAME studiocast-vulkan-qualified-final-executor-tests
                COMMAND studiocast-vulkan-qualified-final-executor-tests)
        studiocast_set_test_labels(studiocast-vulkan-qualified-final-executor-tests
                integration video gpu vulkan)

        add_executable(studiocast-vulkan-matting-runtime-tests
                ${PROJECT_SOURCE_DIR}/tests/gpu/vulkan/vulkan_matting_runtime_tests.cpp
        )

        target_link_libraries(studiocast-vulkan-matting-runtime-tests PRIVATE
                studiocast_vulkan_backend
        )

        target_include_directories(studiocast-vulkan-matting-runtime-tests PRIVATE
                "${PROJECT_SOURCE_DIR}/src"
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
        studiocast_set_test_labels(studiocast-vulkan-matting-runtime-tests
                unit video gpu vulkan model)
    endif()

    add_executable(studiocast-v4l2loopback-diagnostics-tests
            ${PROJECT_SOURCE_DIR}/tests/integration/video/v4l2loopback_diagnostics_tests.cpp
    )

    target_link_libraries(studiocast-v4l2loopback-diagnostics-tests PRIVATE
            studiocast_video_io)

    target_include_directories(studiocast-v4l2loopback-diagnostics-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src"
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
    studiocast_set_test_labels(studiocast-v4l2loopback-diagnostics-tests
            integration video)

    if (_studiocast_have_dlib)
        foreach(_dlib_test IN ITEMS shape-predictor native-oracle)
            string(REPLACE "-" "_" _dlib_test_file "${_dlib_test}")
            set(_dlib_target "studiocast-prepared-cuda-dlib-${_dlib_test}-tests")
            add_executable(${_dlib_target}
                    ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_cuda_dlib_${_dlib_test_file}_tests.cpp)
            target_link_libraries(${_dlib_target} PRIVATE studiocast_open_video_backend)
            if (TARGET dlib::dlib)
                target_link_libraries(${_dlib_target} PRIVATE dlib::dlib)
            elseif (TARGET dlib)
                target_link_libraries(${_dlib_target} PRIVATE dlib)
            endif ()
            studiocast_set_project_warnings(${_dlib_target})
            studiocast_enable_sanitizers(${_dlib_target})
            add_test(NAME ${_dlib_target} COMMAND ${_dlib_target})
            studiocast_set_test_labels(${_dlib_target} integration video gpu model)
        endforeach ()
        set_tests_properties(studiocast-prepared-cuda-dlib-native-oracle-tests
                PROPERTIES SKIP_RETURN_CODE 77)
    endif ()

    add_executable(studiocast-prepared-cuda-eye-contact-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_cuda_eye_contact_tests.cpp)
    target_link_libraries(studiocast-prepared-cuda-eye-contact-tests PRIVATE
            studiocast_open_video_backend)
    studiocast_set_project_warnings(studiocast-prepared-cuda-eye-contact-tests)
    studiocast_enable_sanitizers(studiocast-prepared-cuda-eye-contact-tests)
    add_test(NAME studiocast-prepared-cuda-eye-contact-tests
            COMMAND studiocast-prepared-cuda-eye-contact-tests)
    studiocast_set_test_labels(studiocast-prepared-cuda-eye-contact-tests
            integration video gpu model)

    add_executable(studiocast-prepared-cuda-auto-frame-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_cuda_auto_frame_tests.cpp)
    target_link_libraries(studiocast-prepared-cuda-auto-frame-tests PRIVATE
            studiocast_open_video_backend)
    studiocast_set_project_warnings(studiocast-prepared-cuda-auto-frame-tests)
    studiocast_enable_sanitizers(studiocast-prepared-cuda-auto-frame-tests)
    add_test(NAME studiocast-prepared-cuda-auto-frame-tests
            COMMAND studiocast-prepared-cuda-auto-frame-tests)
    studiocast_set_test_labels(studiocast-prepared-cuda-auto-frame-tests
            integration video gpu model)

    add_executable(studiocast-prepared-cuda-yunet-detector-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_cuda_yunet_detector_tests.cpp)
    target_link_libraries(studiocast-prepared-cuda-yunet-detector-tests PRIVATE
            studiocast_open_video_backend)
    studiocast_set_project_warnings(studiocast-prepared-cuda-yunet-detector-tests)
    studiocast_enable_sanitizers(studiocast-prepared-cuda-yunet-detector-tests)
    add_test(NAME studiocast-prepared-cuda-yunet-detector-tests
            COMMAND studiocast-prepared-cuda-yunet-detector-tests)
    studiocast_set_test_labels(studiocast-prepared-cuda-yunet-detector-tests
            integration video gpu model)

    add_executable(studiocast-prepared-cuda-fastdvdnet-denoiser-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/open_video/prepared_cuda_fastdvdnet_denoiser_tests.cpp)
    target_link_libraries(studiocast-prepared-cuda-fastdvdnet-denoiser-tests PRIVATE
            studiocast_open_video_backend)
    studiocast_set_project_warnings(studiocast-prepared-cuda-fastdvdnet-denoiser-tests)
    studiocast_enable_sanitizers(studiocast-prepared-cuda-fastdvdnet-denoiser-tests)
    add_test(NAME studiocast-prepared-cuda-fastdvdnet-denoiser-tests
            COMMAND studiocast-prepared-cuda-fastdvdnet-denoiser-tests)
    studiocast_set_test_labels(studiocast-prepared-cuda-fastdvdnet-denoiser-tests
            integration video gpu model)

    add_executable(studiocast-cuda-prepared-matte-bounds-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_matte_bounds_tests.cpp)
    target_link_libraries(studiocast-cuda-prepared-matte-bounds-tests PRIVATE
            studiocast_cuda_support)
    target_include_directories(studiocast-cuda-prepared-matte-bounds-tests PRIVATE
            ${PROJECT_SOURCE_DIR}/src)
    studiocast_set_project_warnings(studiocast-cuda-prepared-matte-bounds-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-matte-bounds-tests)
    add_test(NAME studiocast-cuda-prepared-matte-bounds-tests
            COMMAND studiocast-cuda-prepared-matte-bounds-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-matte-bounds-tests
            unit video gpu)
    add_test(NAME studiocast-cuda-prepared-matte-bounds-native-tests
            COMMAND studiocast-cuda-prepared-matte-bounds-tests --native)
    set_tests_properties(studiocast-cuda-prepared-matte-bounds-native-tests PROPERTIES
            SKIP_RETURN_CODE 77)
    studiocast_set_test_labels(studiocast-cuda-prepared-matte-bounds-native-tests
            integration video gpu)

    add_executable(studiocast-cuda-prepared-mirror-kernels-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_mirror_tests.cpp)
    target_link_libraries(studiocast-cuda-prepared-mirror-kernels-tests PRIVATE
            studiocast_cuda_support)
    target_include_directories(studiocast-cuda-prepared-mirror-kernels-tests PRIVATE
            ${PROJECT_SOURCE_DIR}/src)
    studiocast_set_project_warnings(studiocast-cuda-prepared-mirror-kernels-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-mirror-kernels-tests)
    add_test(NAME studiocast-cuda-prepared-mirror-kernels-tests
            COMMAND studiocast-cuda-prepared-mirror-kernels-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-mirror-kernels-tests
            unit video gpu)
    add_test(NAME studiocast-cuda-prepared-mirror-native-tests
            COMMAND studiocast-cuda-prepared-mirror-kernels-tests --native)
    set_tests_properties(studiocast-cuda-prepared-mirror-native-tests PROPERTIES
            SKIP_RETURN_CODE 77)
    studiocast_set_test_labels(studiocast-cuda-prepared-mirror-native-tests
            integration video gpu)

    add_executable(studiocast-cuda-prepared-eye-contact-kernels-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_eye_contact_kernels_tests.cpp)
    target_link_libraries(studiocast-cuda-prepared-eye-contact-kernels-tests PRIVATE
            studiocast_cuda_support)
    target_include_directories(studiocast-cuda-prepared-eye-contact-kernels-tests PRIVATE
            ${PROJECT_SOURCE_DIR}/src)
    studiocast_set_project_warnings(studiocast-cuda-prepared-eye-contact-kernels-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-eye-contact-kernels-tests)
    add_test(NAME studiocast-cuda-prepared-eye-contact-kernels-tests
            COMMAND studiocast-cuda-prepared-eye-contact-kernels-tests)
    studiocast_set_test_labels(studiocast-cuda-prepared-eye-contact-kernels-tests
            unit video gpu)
    add_test(NAME studiocast-cuda-prepared-eye-contact-native-tests
            COMMAND studiocast-cuda-prepared-eye-contact-kernels-tests --native)
    set_tests_properties(studiocast-cuda-prepared-eye-contact-native-tests PROPERTIES
            SKIP_RETURN_CODE 77)
    studiocast_set_test_labels(studiocast-cuda-prepared-eye-contact-native-tests
            integration video gpu)

    add_executable(studiocast-cuda-prepared-fastdvdnet-kernels-tests
            ${PROJECT_SOURCE_DIR}/tests/gpu/cuda/prepared_fastdvdnet_kernels_tests.cpp)
    target_link_libraries(studiocast-cuda-prepared-fastdvdnet-kernels-tests PRIVATE
            studiocast_cuda_support)
    target_include_directories(studiocast-cuda-prepared-fastdvdnet-kernels-tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src" "${STUDIOCAST_GENERATED_DIR}")
    studiocast_set_project_warnings(studiocast-cuda-prepared-fastdvdnet-kernels-tests)
    studiocast_enable_sanitizers(studiocast-cuda-prepared-fastdvdnet-kernels-tests)
    set_target_properties(studiocast-cuda-prepared-fastdvdnet-kernels-tests PROPERTIES
            AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    add_test(NAME studiocast-cuda-prepared-fastdvdnet-kernels-tests
            COMMAND studiocast-cuda-prepared-fastdvdnet-kernels-tests)
    set_tests_properties(studiocast-cuda-prepared-fastdvdnet-kernels-tests PROPERTIES
            SKIP_RETURN_CODE 77)
    studiocast_set_test_labels(studiocast-cuda-prepared-fastdvdnet-kernels-tests
            integration gpu)

    add_test(NAME studiocast-ctest-label-metadata-tests
            COMMAND "${CMAKE_COMMAND}"
                    "-DCTEST_EXECUTABLE=${CMAKE_CTEST_COMMAND}"
                    "-DCTEST_BUILD_DIR=${PROJECT_BINARY_DIR}"
                    -P "${PROJECT_SOURCE_DIR}/tests/support/validate_ctest_labels.cmake")
    studiocast_set_test_labels(studiocast-ctest-label-metadata-tests unit core)
endif()
