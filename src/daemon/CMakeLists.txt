# -----------------------------
# Background daemon
# -----------------------------
add_library(studiocast_daemon_status STATIC
        ${PROJECT_SOURCE_DIR}/src/daemon/status_json.cpp
)

target_link_libraries(studiocast_daemon_status PUBLIC
        studiocast_video_execution_runtime)

target_include_directories(studiocast_daemon_status PUBLIC
        "${PROJECT_SOURCE_DIR}/src"
)

studiocast_set_project_warnings(studiocast_daemon_status)
studiocast_enable_sanitizers(studiocast_daemon_status)

set_target_properties(studiocast_daemon_status PROPERTIES
        AUTOMOC OFF
        AUTOUIC OFF
        AUTORCC OFF
)

studiocast_assert_direct_dependencies(studiocast_daemon_status
        PUBLIC_STUDIOCAST_DEPS studiocast_video_execution_runtime)
studiocast_assert_neutral_target(studiocast_daemon_status)

add_executable(studiocastd
        ${PROJECT_SOURCE_DIR}/src/daemon/studiocastd_main.cpp
)

target_link_libraries(studiocastd PRIVATE
        studiocast_daemon_status
        studiocast_ipc
        studiocast_runtime_config)

target_include_directories(studiocastd PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
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
