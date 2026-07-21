include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

function(studiocast_configure_onnxruntime
    out_found out_target out_has_cuda_ep_v2 out_has_tensorrt_ep_v2)
  set(_found FALSE)
  set(_target "")

  # 1) Prefer a CMake package config if available.
  find_package(onnxruntime CONFIG QUIET)
  if (onnxruntime_FOUND)
    if (TARGET onnxruntime::onnxruntime)
      set(_found TRUE)
      set(_target onnxruntime::onnxruntime)
    elseif (TARGET onnxruntime::onnxruntime_shared)
      set(_found TRUE)
      set(_target onnxruntime::onnxruntime_shared)
    elseif (TARGET onnxruntime::onnxruntime_static)
      set(_found TRUE)
      set(_target onnxruntime::onnxruntime_static)
    elseif (TARGET onnxruntime)
      set(_found TRUE)
      set(_target onnxruntime)
    elseif (TARGET onnxruntime_shared)
      set(_found TRUE)
      set(_target onnxruntime_shared)
    elseif (TARGET onnxruntime_static)
      set(_found TRUE)
      set(_target onnxruntime_static)
    endif()
  endif()

  # 2) Fall back to pkg-config (common for distro packages).
  if (NOT _found)
    if (PkgConfig_FOUND)
      pkg_check_modules(ONNXRUNTIME QUIET IMPORTED_TARGET onnxruntime)
      if (ONNXRUNTIME_FOUND)
        # NOTE: Some environments can have stale/broken pkg-config metadata that
        # points to non-existent include directories. CMake treats missing paths
        # in INTERFACE_INCLUDE_DIRECTORIES as a hard error during generation.
        #
        # Sanitize the imported target and only accept it if headers are actually
        # present.
        set(_ort_pkg_includes_ok "")
        set(_ort_pkg_includes_bad "")
        set(_ort_pkg_has_header FALSE)

        get_target_property(_ort_pkg_includes PkgConfig::ONNXRUNTIME INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT _ort_pkg_includes)
          set(_ort_pkg_includes "")
        endif()

        foreach (d IN LISTS _ort_pkg_includes)
          if (d MATCHES "\\$<")
            list(APPEND _ort_pkg_includes_ok "${d}")
          elseif (EXISTS "${d}")
            list(APPEND _ort_pkg_includes_ok "${d}")
            if (EXISTS "${d}/onnxruntime_cxx_api.h")
              set(_ort_pkg_has_header TRUE)
            endif()
          else()
            list(APPEND _ort_pkg_includes_bad "${d}")
          endif()
        endforeach()

        if (_ort_pkg_includes_bad)
          set_target_properties(PkgConfig::ONNXRUNTIME PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_ort_pkg_includes_ok}"
          )
        endif()

        if (_ort_pkg_has_header)
          set(_found TRUE)
          set(_target PkgConfig::ONNXRUNTIME)
        else()
          message(STATUS "ONNX Runtime found via pkg-config but headers were not usable; ignoring pkg-config result. Install onnxruntime dev package or set ONNXRUNTIME_ROOT.")
          set(_found FALSE)
          set(_target "")
        endif()
      endif()
    endif()
  endif()

  

  # 3) Dev convenience: try to locate ONNX Runtime from an installed Python package
  # (onnxruntime / onnxruntime-gpu). This is useful when a developer installed ORT via pip
  # but hasn't installed the standalone C/C++ SDK.
  #
  # Note: This is best-effort and only used as a fallback. For system installs / packaging,
  # prefer a proper CMake package, pkg-config, or ONNXRUNTIME_ROOT.
  if (NOT _found)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
      set(_studiocast_ort_py_probe [==[
import json
import pathlib
import sys

try:
    import onnxruntime  # noqa: F401
except Exception:
    sys.exit(2)

root = pathlib.Path(onnxruntime.__file__).resolve().parent

# Find headers (need onnxruntime_cxx_api.h).
inc = None
for p in [
    root / "capi" / "include",
    root / "capi" / "include" / "onnxruntime",
    root / "include",
    root / "include" / "onnxruntime",
    root / "capi",
]:
    if (p / "onnxruntime_cxx_api.h").exists():
        inc = p
        break

if inc is None:
    for hdr in root.rglob("onnxruntime_cxx_api.h"):
        inc = hdr.parent
        break

# Find libonnxruntime.
lib = None
for d in [
    root / "capi",
    root / "capi" / "lib",
    root / "lib",
    root,
]:
    if not d.exists():
        continue

    p = d / "libonnxruntime.so"
    if p.exists():
        lib = p
        break

    # Fall back to a versioned .so if the unversioned symlink isn't present.
    candidates = sorted(d.glob("libonnxruntime.so.*"))
    if candidates:
        lib = candidates[-1]
        break

if not inc or not lib:
    sys.exit(3)

payload = {
    "root": str(root),
    "include": str(inc),
    "lib": str(lib),
    "libdir": str(lib.parent),
}
print(json.dumps(payload))
]==])

      execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "${_studiocast_ort_py_probe}"
        RESULT_VARIABLE _studiocast_ort_py_rc
        OUTPUT_VARIABLE _studiocast_ort_py_out
        ERROR_VARIABLE _studiocast_ort_py_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      if (_studiocast_ort_py_rc EQUAL 0)
        # Parse JSON payload.
        set(_studiocast_ort_py_json "${_studiocast_ort_py_out}")
        string(JSON _studiocast_ort_py_inc GET "${_studiocast_ort_py_json}" include)
        string(JSON _studiocast_ort_py_lib GET "${_studiocast_ort_py_json}" lib)
        string(JSON _studiocast_ort_py_libdir GET "${_studiocast_ort_py_json}" libdir)

        if (EXISTS "${_studiocast_ort_py_inc}/onnxruntime_cxx_api.h" AND EXISTS "${_studiocast_ort_py_lib}")
          if (NOT TARGET studiocast_onnxruntime)
            add_library(studiocast_onnxruntime UNKNOWN IMPORTED)
            set_target_properties(studiocast_onnxruntime PROPERTIES
              IMPORTED_LOCATION "${_studiocast_ort_py_lib}"
              INTERFACE_INCLUDE_DIRECTORIES "${_studiocast_ort_py_inc}"
            )

            # Ensure consumers can run without needing LD_LIBRARY_PATH when ORT lives in
            # a non-system directory (e.g., ~/.local from pip). Many ORT builds ship
            # provider .so's next to libonnxruntime.
            if (UNIX AND NOT APPLE)
              set_property(TARGET studiocast_onnxruntime PROPERTY
                INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_studiocast_ort_py_libdir}"
              )
            endif()
          endif()
          if (NOT TARGET studiocast::onnxruntime)
            add_library(studiocast::onnxruntime ALIAS studiocast_onnxruntime)
          endif()

          set(_found TRUE)
          set(_target studiocast::onnxruntime)
          message(STATUS "ONNX Runtime found via Python package (${Python3_EXECUTABLE}): ${_studiocast_ort_py_lib}")
        endif()
      endif()
    endif()
  endif()

  # 4) Last resort: user-provided root path.
  if (NOT _found AND DEFINED ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      HINTS "${ONNXRUNTIME_ROOT}"
      PATH_SUFFIXES include include/onnxruntime
    )

    find_library(ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      HINTS "${ONNXRUNTIME_ROOT}"
      PATH_SUFFIXES lib lib64
    )

    if (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
      if (NOT TARGET studiocast_onnxruntime)
        add_library(studiocast_onnxruntime UNKNOWN IMPORTED)
        set_target_properties(studiocast_onnxruntime PROPERTIES
          IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
      endif()
      if (NOT TARGET studiocast::onnxruntime)
        add_library(studiocast::onnxruntime ALIAS studiocast_onnxruntime)
      endif()

      set(_found TRUE)
      set(_target studiocast::onnxruntime)
    endif()
  endif()

  # Feature probe: CUDA EP V2 provider options.
  # We want to use OrtApi::CreateCUDAProviderOptions + SessionOptionsAppendExecutionProvider_CUDA_V2
  # where headers support it, and keep compatibility where they don't.
  set(_has_cuda_ep_v2 FALSE)
  if (_found)
    set(_probe_src [==[
#include <onnxruntime_cxx_api.h>

int main() {
  auto& api = Ort::GetApi();

  OrtCUDAProviderOptionsV2* opts = nullptr;
  api.CreateCUDAProviderOptions(&opts);

  Ort::SessionOptions so;
  api.SessionOptionsAppendExecutionProvider_CUDA_V2(so, opts);
  api.ReleaseCUDAProviderOptions(opts);
  return 0;
}
]==])

    # Best-effort propagation of include dirs / compile defs from the chosen target.
    # check_cxx_source_compiles does not automatically add target usage requirements.
    get_target_property(_ort_includes "${_target}" INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT _ort_includes)
      set(_ort_includes "")
    endif()
    get_target_property(_ort_compile_defs "${_target}" INTERFACE_COMPILE_DEFINITIONS)
    if (NOT _ort_compile_defs)
      set(_ort_compile_defs "")
    endif()

    set(_ort_includes_sanitized "")
    foreach (d IN LISTS _ort_includes)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ort_includes_sanitized "${d}")
      endif()
    endforeach()

    set(_ort_compile_defs_sanitized "")
    foreach (d IN LISTS _ort_compile_defs)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ort_compile_defs_sanitized "${d}")
      endif()
    endforeach()

    set(_old_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_old_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
    set(_old_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")

    set(CMAKE_REQUIRED_INCLUDES "${_ort_includes_sanitized}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_ort_compile_defs_sanitized}")
    # Compile-only feature probe: avoid link failures when only headers are present.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    check_cxx_source_compiles("${_probe_src}" _studiocast_ort_cuda_ep_v2_probe_ok)

    set(CMAKE_REQUIRED_INCLUDES "${_old_required_includes}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_old_required_definitions}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_old_try_compile_target_type}")

    if (_studiocast_ort_cuda_ep_v2_probe_ok)
      set(_has_cuda_ep_v2 TRUE)
    endif()
  endif()

  if (_has_cuda_ep_v2)
    message(STATUS "ONNX Runtime CUDA EP V2 provider options: available (STUDIOCAST_ORT_HAS_CUDA_EP_V2=1)")
  else()
    if (_found)
      message(STATUS "ONNX Runtime CUDA EP V2 provider options: unavailable (STUDIOCAST_ORT_HAS_CUDA_EP_V2=0)")
    else()
      message(STATUS "ONNX Runtime CUDA EP V2 provider options: unavailable (onnxruntime not found; STUDIOCAST_ORT_HAS_CUDA_EP_V2=0)")
    endif()
  endif()

  # Feature probe: TensorRT EP V2 provider options.
  # TensorRT support is optional even for GPU ONNX Runtime builds. Probe the
  # headers/API surface and let runtime provider availability decide whether an
  # actual session can append TensorRT.
  set(_has_tensorrt_ep_v2 FALSE)
  if (_found)
    set(_probe_src [==[
#include <onnxruntime_cxx_api.h>

int main() {
  auto& api = Ort::GetApi();

  OrtTensorRTProviderOptionsV2* opts = nullptr;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&opts));

  const char* keys[] = {"device_id"};
  const char* values[] = {"0"};
  Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(opts, keys, values, 1));
  void* stream = nullptr;
  Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(opts, "user_compute_stream", stream));

  Ort::SessionOptions so;
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(so, opts));
  api.ReleaseTensorRTProviderOptions(opts);
  return 0;
}
]==])

    # Best-effort propagation of include dirs / compile defs from the chosen target.
    # check_cxx_source_compiles does not automatically add target usage requirements.
    get_target_property(_ort_includes "${_target}" INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT _ort_includes)
      set(_ort_includes "")
    endif()
    get_target_property(_ort_compile_defs "${_target}" INTERFACE_COMPILE_DEFINITIONS)
    if (NOT _ort_compile_defs)
      set(_ort_compile_defs "")
    endif()

    set(_ort_includes_sanitized "")
    foreach (d IN LISTS _ort_includes)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ort_includes_sanitized "${d}")
      endif()
    endforeach()

    set(_ort_compile_defs_sanitized "")
    foreach (d IN LISTS _ort_compile_defs)
      if (NOT d MATCHES "\\$<")
        list(APPEND _ort_compile_defs_sanitized "${d}")
      endif()
    endforeach()

    set(_old_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_old_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
    set(_old_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")

    set(CMAKE_REQUIRED_INCLUDES "${_ort_includes_sanitized}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_ort_compile_defs_sanitized}")
    # Compile-only feature probe: avoid link failures when only headers are present.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    check_cxx_source_compiles("${_probe_src}" _studiocast_ort_tensorrt_ep_v2_probe_ok)

    set(CMAKE_REQUIRED_INCLUDES "${_old_required_includes}")
    set(CMAKE_REQUIRED_DEFINITIONS "${_old_required_definitions}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_old_try_compile_target_type}")

    if (_studiocast_ort_tensorrt_ep_v2_probe_ok)
      set(_has_tensorrt_ep_v2 TRUE)
    endif()
  endif()

  if (_has_tensorrt_ep_v2)
    message(STATUS "ONNX Runtime TensorRT EP V2 provider options: available (STUDIOCAST_ORT_HAS_TENSORRT_EP_V2=1)")
  else()
    if (_found)
      message(STATUS "ONNX Runtime TensorRT EP V2 provider options: unavailable (STUDIOCAST_ORT_HAS_TENSORRT_EP_V2=0)")
    else()
      message(STATUS "ONNX Runtime TensorRT EP V2 provider options: unavailable (onnxruntime not found; STUDIOCAST_ORT_HAS_TENSORRT_EP_V2=0)")
    endif()
  endif()

  set(${out_found} ${_found} PARENT_SCOPE)
  set(${out_target} "${_target}" PARENT_SCOPE)
  set(${out_has_cuda_ep_v2} ${_has_cuda_ep_v2} PARENT_SCOPE)
  set(${out_has_tensorrt_ep_v2} ${_has_tensorrt_ep_v2} PARENT_SCOPE)
endfunction()
