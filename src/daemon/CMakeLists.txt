# -----------------------------
# Background daemon
# -----------------------------
add_executable(studiocastd
        ${PROJECT_SOURCE_DIR}/src/daemon/studiocastd_main.cpp
)

target_link_libraries(studiocastd PRIVATE studiocast_core)

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
