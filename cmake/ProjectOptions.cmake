include_guard(GLOBAL)

option(STUDIOCAST_ENABLE_WERROR "Treat warnings as errors" OFF)
option(STUDIOCAST_ENABLE_SANITIZERS "Enable sanitizers (ASan/UBSan) on supported compilers" OFF)
option(STUDIOCAST_ENABLE_LTO "Enable link-time optimization (IPO/LTO) if supported" OFF)
option(STUDIOCAST_ENABLE_CUDA_KERNELS "Build optional CUDA .cu kernels (requires CUDA toolkit)" OFF)
option(STUDIOCAST_BUILD_BENCHMARKS "Build developer benchmark tools" OFF)

set(_studiocast_default_open_cuda OFF)
if(UNIX AND NOT APPLE)
  set(_studiocast_default_open_cuda ON)
endif()
option(STUDIOCAST_ENABLE_OPEN_CUDA "Enable Open CUDA backend (requires ONNX Runtime + CUDA EP)" ${_studiocast_default_open_cuda})

option(STUDIOCAST_ENABLE_OPEN_VULKAN "Enable Open Vulkan backend (runtime-loaded Vulkan compute)" OFF)
option(STUDIOCAST_ENABLE_NCNN_SPIKE "Build experimental ncnn Vulkan matting spike tool" OFF)
option(STUDIOCAST_REQUIRE_NCNN "Fail configure when the ncnn spike is enabled but ncnn is unavailable" OFF)
option(STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
       "Enable ncnn Vulkan matting build prerequisites (does not imply production readiness; requires Open Vulkan and Vulkan-enabled ncnn)"
       OFF)

set(_studiocast_default_open_audio OFF)
if(UNIX AND NOT APPLE)
  set(_studiocast_default_open_audio ON)
endif()
option(STUDIOCAST_ENABLE_OPEN_AUDIO "Enable Open Audio backend (requires ONNX Runtime; CPU EP baseline)" ${_studiocast_default_open_audio})

# Optional dependency used for Open Video Eye Contact (face landmarks via dlib).
#
# If disabled or dlib is not found, the open-source Eye Contact backend will be
# unavailable (Maxine AR may still provide Eye Contact when present).
option(STUDIOCAST_ENABLE_DLIB "Enable dlib support (face landmarks)" ON)

function(studiocast_setup_options)
  # Intentionally light for now; expand later
endfunction()

function(studiocast_global_options)
  set(CMAKE_CXX_STANDARD 20 PARENT_SCOPE)
  set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
  set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)

  # Helpful for clangd/CLion indexing
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)

  if(STUDIOCAST_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_error)
    if(_ipo_supported)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE PARENT_SCOPE)
    else()
      message(WARNING "IPO/LTO not supported: ${_ipo_error}")
    endif()
  endif()
endfunction()
