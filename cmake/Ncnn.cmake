include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

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

#ifndef NCNN_VULKAN
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

  set(${out_found} ${_found} PARENT_SCOPE)
  set(${out_target} "${_target}" PARENT_SCOPE)
  set(${out_has_vulkan} ${_has_vulkan} PARENT_SCOPE)
endfunction()
