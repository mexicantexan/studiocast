include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

function(studiocast_install_ncnn_runtime_notices)
  if (NOT STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING)
    return()
  endif()

  # The reviewed helper keeps notices beside the selected package, never in a
  # separately selected SDK whose receipt could describe different linked bytes.
  if (ncnn_DIR)
    get_filename_component(_prefix "${ncnn_DIR}/../../.." ABSOLUTE)
  elseif (NCNN_ROOT AND _studiocast_ncnn_target STREQUAL "studiocast::ncnn")
    get_filename_component(_prefix "${NCNN_ROOT}" ABSOLUTE)
  else()
    message(FATAL_ERROR "Production ncnn Runtime packaging requires the prefix from scripts/build_ncnn_vulkan_dependency.py; set ncnn_DIR or CMAKE_PREFIX_PATH to that prefix.")
  endif()
  set(_notices "${_prefix}/share/licenses/studiocast-ncnn")
  set(_files ncnn-LICENSE.txt glslang-LICENSE.txt dependency.json)
  foreach (_file IN LISTS _files)
    if (NOT EXISTS "${_notices}/${_file}" OR IS_DIRECTORY "${_notices}/${_file}")
      message(FATAL_ERROR "Production ncnn Runtime packaging requires ${_notices}/${_file}. Rebuild the isolated dependency with scripts/build_ncnn_vulkan_dependency.py.")
    endif()
    file(SIZE "${_notices}/${_file}" _size)
    if (_size EQUAL 0)
      message(FATAL_ERROR "Production ncnn Runtime packaging requires nonempty ${_notices}/${_file}.")
    endif()
  endforeach()
  file(READ "${_notices}/dependency.json" _receipt)
  file(SHA256 "${PROJECT_SOURCE_DIR}/cmake/patches/ncnn-studiocast-replay.patch" _patch_sha256)
  set(_upstream_revision "24ea07e87f544a3fbc59980b57885a92c75f2d24")
  set(_glslang_revision "5017667d5a5300dff6aab57db1931a495ff9a10c")
  set(_replay_abi 2)
  foreach (_field IN ITEMS upstream_revision glslang_revision replay_abi patch_sha256)
    string(JSON _value ERROR_VARIABLE _error GET "${_receipt}" "${_field}")
    if (_error OR NOT "${_value}" STREQUAL "${_${_field}}")
      message(FATAL_ERROR "Production ncnn Runtime packaging dependency receipt has invalid ${_field}: ${_notices}/dependency.json. Rebuild with the current scripts/build_ncnn_vulkan_dependency.py.")
    endif()
  endforeach()

  # Snapshot the validated inputs so the installed receipt and licenses stay
  # together even if the SDK prefix is subsequently moved or removed.
  set(_stage "${CMAKE_CURRENT_BINARY_DIR}/ncnn-runtime-notices")
  foreach (_file IN LISTS _files)
    configure_file("${_notices}/${_file}" "${_stage}/${_file}" COPYONLY)
    install(FILES "${_stage}/${_file}"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/studiocast-ncnn"
      COMPONENT Runtime)
  endforeach()
endfunction()

function(studiocast_configure_ncnn out_found out_target out_has_vulkan)
  set(_found FALSE)
  set(_target "")
  set(_has_vulkan FALSE)

  find_package(ncnn CONFIG QUIET)
  if (ncnn_FOUND)
    if (TARGET ncnn)
      set(_found TRUE)
      set(_target ncnn)
    elseif (TARGET ncnn::ncnn)
      set(_found TRUE)
      set(_target ncnn::ncnn)
    endif()
  endif()

  if (NOT _found)
    if (PkgConfig_FOUND)
      pkg_check_modules(NCNN QUIET IMPORTED_TARGET ncnn)
      if (NCNN_FOUND AND TARGET PkgConfig::NCNN)
        set(_found TRUE)
        set(_target PkgConfig::NCNN)
      endif()
    endif()
  endif()

  if (NOT _found AND DEFINED NCNN_ROOT)
    find_path(NCNN_INCLUDE_DIR
      NAMES net.h ncnn/net.h
      HINTS "${NCNN_ROOT}"
      PATH_SUFFIXES include include/ncnn
    )
    find_library(NCNN_LIBRARY
      NAMES ncnn
      HINTS "${NCNN_ROOT}"
      PATH_SUFFIXES lib lib64
    )
    if (NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
      if (NOT TARGET studiocast_ncnn)
        add_library(studiocast_ncnn UNKNOWN IMPORTED)
        set_target_properties(studiocast_ncnn PROPERTIES
          IMPORTED_LOCATION "${NCNN_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}"
        )
      endif()
      if (NOT TARGET studiocast::ncnn)
        add_library(studiocast::ncnn ALIAS studiocast_ncnn)
      endif()
      set(_found TRUE)
      set(_target studiocast::ncnn)
    endif()
  endif()

  if (_found)
    set(_probe_src [==[
#include <net.h>
#include <gpu.h>

#if !defined(NCNN_VULKAN) || NCNN_VULKAN != 1
#error "ncnn was not built with Vulkan support"
#endif

int main() {
  ncnn::Net net;
  net.opt.use_vulkan_compute = true;
  return 0;
}
]==])

    get_target_property(_ncnn_includes "${_target}" INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT _ncnn_includes)
      set(_ncnn_includes "")
    endif()
    get_target_property(_ncnn_defs "${_target}" INTERFACE_COMPILE_DEFINITIONS)
    if (NOT _ncnn_defs)
      set(_ncnn_defs "")
    endif()

    set(_ncnn_includes_sanitized "")
    foreach (d IN LISTS _ncnn_includes)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ncnn_includes_sanitized "${d}")
      endif()
    endforeach()

    set(_ncnn_defs_sanitized "")
    foreach (d IN LISTS _ncnn_defs)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ncnn_defs_sanitized "-D${d}")
      endif()
    endforeach()

    set(_old_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_old_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
    set(_old_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")

    set(CMAKE_REQUIRED_INCLUDES "${_ncnn_includes_sanitized}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_ncnn_defs_sanitized}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    check_cxx_source_compiles("${_probe_src}" _studiocast_ncnn_vulkan_probe_ok)

    set(CMAKE_REQUIRED_INCLUDES "${_old_required_includes}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_old_required_definitions}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_old_try_compile_target_type}")

    if (_studiocast_ncnn_vulkan_probe_ok)
      set(_has_vulkan TRUE)
    endif()
  endif()

  if (STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING AND _found)
    set(CMAKE_REQUIRED_LIBRARIES "${_target}")
    unset(_studiocast_ncnn_replay_link_ok CACHE)
    check_cxx_source_compiles([==[
#include <net.h>
#include <gpu.h>
#include <command.h>
#include <string_view>
#if !defined(NCNN_VULKAN) || NCNN_VULKAN != 1 || !defined(NCNN_STUDIOCAST_REPLAY_ABI) || NCNN_STUDIOCAST_REPLAY_ABI != 2
#error "Production matting requires pinned StudioCast replay API"
#endif
#if !NCNN_SIMPLEVK
#error "Production adoption requires the pinned SIMPLEVK provider entry point"
#endif
#if NCNN_COVERAGE
#error "Production replay forbids ncnn coverage teardown waits"
#endif
int main() {
  static_assert(std::string_view(NCNN_STUDIOCAST_UPSTREAM_REVISION) == "24ea07e87f544a3fbc59980b57885a92c75f2d24");
  if (ncnn::studiocast_acquire_gpu_instance() != 0) return 1;
  ncnn::VulkanDevice device(0, true);
  ncnn::VkCompute compute(&device, true);
  ncnn::VkTransfer transfer(&device, true);
  VkCommandBuffer buffer = VK_NULL_HANDLE;
  ncnn::Option option;
  option.studiocast_model_upload = &transfer;
  int result = compute.studiocast_finalize(&buffer) + transfer.studiocast_finalize(&buffer) + device.studiocast_initialize_dummy(&buffer);
  result += compute.studiocast_valid() && transfer.studiocast_valid() ? 0 : 1;
  result += static_cast<int>(device.studiocast_dummy_allocation_bytes() + device.studiocast_dummy_allocation_count());
  ncnn::studiocast_release_gpu_instance();
  return result;
}
]==] _studiocast_ncnn_replay_link_ok)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if (NOT _studiocast_ncnn_replay_link_ok)
      message(FATAL_ERROR "Production ncnn matting requires upstream 24ea07e87f544a3fbc59980b57885a92c75f2d24 with cmake/patches/ncnn-studiocast-replay.patch. Build the isolated dependency with scripts/build_ncnn_vulkan_dependency.py.")
    endif()
  endif()

  set(${out_found} ${_found} PARENT_SCOPE)
  set(${out_target} "${_target}" PARENT_SCOPE)
  set(${out_has_vulkan} ${_has_vulkan} PARENT_SCOPE)
endfunction()
