include_guard(GLOBAL)

function(studiocast_assert_direct_dependencies target)
    set(multi_value_args PUBLIC_STUDIOCAST_DEPS PRIVATE_STUDIOCAST_DEPS)
    cmake_parse_arguments(ARG "" "" "${multi_value_args}" ${ARGN})

    get_target_property(link_deps "${target}" LINK_LIBRARIES)
    get_target_property(interface_deps "${target}" INTERFACE_LINK_LIBRARIES)
    if (NOT link_deps)
        set(link_deps)
    endif ()
    if (NOT interface_deps)
        set(interface_deps)
    endif ()

    set(actual_studiocast_deps)
    foreach (dependency IN LISTS link_deps)
        if (dependency MATCHES "^studiocast_")
            list(APPEND actual_studiocast_deps "${dependency}")
        endif ()
    endforeach ()
    set(expected_studiocast_deps
            ${ARG_PUBLIC_STUDIOCAST_DEPS}
            ${ARG_PRIVATE_STUDIOCAST_DEPS})
    list(REMOVE_DUPLICATES actual_studiocast_deps)
    list(REMOVE_DUPLICATES expected_studiocast_deps)
    list(SORT actual_studiocast_deps)
    list(SORT expected_studiocast_deps)
    if (NOT "${actual_studiocast_deps}" STREQUAL "${expected_studiocast_deps}")
        message(FATAL_ERROR
                "${target} has forbidden or missing StudioCast dependencies. "
                "Expected '${expected_studiocast_deps}', got '${actual_studiocast_deps}'.")
    endif ()

    set(actual_public_studiocast_deps)
    foreach (dependency IN LISTS interface_deps)
        if (dependency MATCHES "^studiocast_")
            list(APPEND actual_public_studiocast_deps "${dependency}")
        endif ()
    endforeach ()
    list(REMOVE_DUPLICATES actual_public_studiocast_deps)
    list(REMOVE_DUPLICATES ARG_PUBLIC_STUDIOCAST_DEPS)
    list(SORT actual_public_studiocast_deps)
    list(SORT ARG_PUBLIC_STUDIOCAST_DEPS)
    if (NOT "${actual_public_studiocast_deps}" STREQUAL "${ARG_PUBLIC_STUDIOCAST_DEPS}")
        message(FATAL_ERROR
                "${target} has an invalid public StudioCast link interface. "
                "Expected '${ARG_PUBLIC_STUDIOCAST_DEPS}', got '${actual_public_studiocast_deps}'.")
    endif ()
endfunction()

function(studiocast_assert_neutral_target target)
    get_target_property(link_deps "${target}" LINK_LIBRARIES)
    get_target_property(interface_deps "${target}" INTERFACE_LINK_LIBRARIES)
    get_target_property(compile_definitions "${target}" COMPILE_DEFINITIONS)
    string(JOIN ";" architecture_surface
            "${link_deps}" "${interface_deps}" "${compile_definitions}")
    if (architecture_surface MATCHES
            "Qt|PULSE|JPEG|PNG|[Oo][Nn][Nn][Xx]|dlib|ncnn|CUDA::|Vulkan::|Maxine")
        message(FATAL_ERROR
                "Neutral target ${target} leaks an optional SDK, GUI, image, or service dependency: ${architecture_surface}")
    endif ()
    if (architecture_surface MATCHES "STUDIOCAST_(ENABLE|HAVE)_")
        message(FATAL_ERROR
                "Neutral target ${target} must not carry feature-availability definitions: ${architecture_surface}")
    endif ()
endfunction()

function(studiocast_assert_cpp_source_ownership)
    set(one_value_args EXPECTED_COUNT)
    set(multi_value_args TARGETS EXPECTED_SOURCES)
    cmake_parse_arguments(ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(owned_sources)
    foreach (target IN LISTS ARG_TARGETS)
        get_target_property(target_sources "${target}" SOURCES)
        foreach (source IN LISTS target_sources)
            if (source MATCHES "^src/core/.+\\.cpp$")
                if (source IN_LIST owned_sources)
                    message(FATAL_ERROR
                            "Compiled production source has multiple owners: ${source}")
                endif ()
                list(APPEND owned_sources "${source}")
            endif ()
        endforeach ()
    endforeach ()

    list(SORT owned_sources)
    list(SORT ARG_EXPECTED_SOURCES)
    list(LENGTH owned_sources owned_count)
    if (NOT owned_count EQUAL ARG_EXPECTED_COUNT)
        message(FATAL_ERROR
                "Configured production source ownership count is ${owned_count}; expected ${ARG_EXPECTED_COUNT}.")
    endif ()
    if (NOT "${owned_sources}" STREQUAL "${ARG_EXPECTED_SOURCES}")
        message(FATAL_ERROR
                "Configured production source ownership is incomplete or contains an unexpected source.")
    endif ()
endfunction()
