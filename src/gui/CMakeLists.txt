# -----------------------------
# GUI executable
# -----------------------------
add_executable(studiocast
        ${PROJECT_SOURCE_DIR}/src/main.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/theme.h
        ${PROJECT_SOURCE_DIR}/src/gui/theme.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/text_edit_utils.h
        ${PROJECT_SOURCE_DIR}/src/gui/text_edit_utils.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/resources/gui_resources.qrc
        ${PROJECT_SOURCE_DIR}/src/gui/main_window.h
        ${PROJECT_SOURCE_DIR}/src/gui/main_window.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/status/daemon_status_snapshot.h
        ${PROJECT_SOURCE_DIR}/src/gui/status/daemon_status_snapshot.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/status/status_poller.h
        ${PROJECT_SOURCE_DIR}/src/gui/status/status_poller.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/home_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/home_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/engines_models_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/engines_models_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/support_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/support_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/settings_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/settings_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/advanced_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/advanced_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/audio_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/audio_page.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/pages/video_page.h
        ${PROJECT_SOURCE_DIR}/src/gui/pages/video_page.cpp
        ${PROJECT_SOURCE_DIR}/src/core/probe/probe.h
)

target_link_libraries(studiocast PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        studiocast_core
)

target_include_directories(studiocast PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

target_compile_definitions(studiocast PRIVATE
        STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

studiocast_set_project_warnings(studiocast)
studiocast_enable_sanitizers(studiocast)

set_target_properties(studiocast PROPERTIES
        OUTPUT_NAME "studiocast"
)
