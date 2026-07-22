# -----------------------------
# GUI installer executable
# -----------------------------
add_executable(studiocast-installer
        ${PROJECT_SOURCE_DIR}/installer/gui/main.cpp
        ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard.h
        ${PROJECT_SOURCE_DIR}/installer/gui/installer_wizard.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/theme.h
        ${PROJECT_SOURCE_DIR}/src/gui/theme.cpp
        ${PROJECT_SOURCE_DIR}/src/gui/resources/gui_resources.qrc
)
target_link_libraries(studiocast-installer PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
)

target_include_directories(studiocast-installer PRIVATE
        "${PROJECT_SOURCE_DIR}/installer/gui"
        "${PROJECT_SOURCE_DIR}/src"
        "${STUDIOCAST_GENERATED_DIR}"
)

target_compile_definitions(studiocast-installer PRIVATE
        STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH="${PROJECT_SOURCE_DIR}/installer/backend/studiocast-installer-backend"
        STUDIOCAST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

studiocast_set_project_warnings(studiocast-installer)
studiocast_enable_sanitizers(studiocast-installer)

set_target_properties(studiocast-installer PROPERTIES
        OUTPUT_NAME "studiocast-installer"
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

install(PROGRAMS ${PROJECT_SOURCE_DIR}/installer/backend/studiocast-installer-backend
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer"
        COMPONENT Installer
)

install(FILES
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/__init__.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/cli.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/contracts.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/executor.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/facts.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/manifest.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/packaged.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/plan.py
        ${PROJECT_SOURCE_DIR}/installer/backend/studiocast_installer_backend/recommendation.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/studiocast_installer_backend"
        COMPONENT Installer
)

install(FILES
        ${PROJECT_SOURCE_DIR}/installer/privileged/__init__.py
        ${PROJECT_SOURCE_DIR}/installer/privileged/client_contract.py
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/privileged"
        COMPONENT Installer
)

install(FILES
        ${PROJECT_SOURCE_DIR}/installer/release/__init__.py
        ${PROJECT_SOURCE_DIR}/installer/release/release_channel.py
        ${PROJECT_SOURCE_DIR}/installer/release/release-manifest-v1.schema.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/release"
        COMPONENT Installer
)

# Curated model transactions execute from the trusted Installer component, not
# from a target-built or user-selected source tree. Keep the installed relative
# layout expected by model_transactions.py while staging metadata only.
install(PROGRAMS ${PROJECT_SOURCE_DIR}/installer/models/studiocast-model-transaction
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/models"
        COMPONENT Installer
)

install(FILES
        ${PROJECT_SOURCE_DIR}/installer/models/__init__.py
        ${PROJECT_SOURCE_DIR}/installer/models/model_transactions.py
        ${PROJECT_SOURCE_DIR}/installer/models/curated-model-catalog-v1.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/models"
        COMPONENT Installer
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/resources/model_packs/
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/resources/model_packs"
        COMPONENT Installer
        FILES_MATCHING
        PATTERN "model.json"
        PATTERN "LICENSE.txt"
)

# The Installer component carries the committed production trust root. Release
# packaging still requires it explicitly and revalidates the same bytes before
# staging the release-grade AppDir.
install(FILES
        ${PROJECT_SOURCE_DIR}/installer/release/keys/README.md
        ${PROJECT_SOURCE_DIR}/installer/release/keys/studiocast-release-2026.pem
        DESTINATION "${CMAKE_INSTALL_DATADIR}/studiocast/installer/trust/keys"
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        COMPONENT Installer
)
