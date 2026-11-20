cmake_minimum_required(VERSION 3.22)

if (NOT DEFINED CTEST_EXECUTABLE OR CTEST_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "CTEST_EXECUTABLE is required")
endif()
if (NOT DEFINED CTEST_BUILD_DIR OR CTEST_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "CTEST_BUILD_DIR is required")
endif()

execute_process(
        COMMAND "${CTEST_EXECUTABLE}" --test-dir "${CTEST_BUILD_DIR}" --show-only=json-v1
        RESULT_VARIABLE ctest_result
        OUTPUT_VARIABLE ctest_json
        ERROR_VARIABLE ctest_error
)
if (NOT ctest_result EQUAL 0)
    message(FATAL_ERROR
            "Unable to read CTest metadata (${ctest_result}): ${ctest_error}")
endif()

set(level_labels unit integration system)
set(domain_labels
        core installer release model gui audio video ipc gpu vulkan maxine)
set(allowed_labels ${level_labels} ${domain_labels})
set(seen_names)
set(validation_errors)

string(JSON test_count LENGTH "${ctest_json}" tests)
if (test_count EQUAL 0)
    list(APPEND validation_errors "CTest registered no automated tests")
else()
    math(EXPR last_test "${test_count} - 1")
    foreach (test_index RANGE 0 ${last_test})
        string(JSON test_name GET "${ctest_json}" tests ${test_index} name)

        list(FIND seen_names "${test_name}" duplicate_index)
        if (NOT duplicate_index EQUAL -1)
            list(APPEND validation_errors "duplicate test name: ${test_name}")
        endif()
        list(APPEND seen_names "${test_name}")

        set(test_labels)
        string(JSON property_count LENGTH
                "${ctest_json}" tests ${test_index} properties)
        if (property_count GREATER 0)
            math(EXPR last_property "${property_count} - 1")
            foreach (property_index RANGE 0 ${last_property})
                string(JSON property_name GET
                        "${ctest_json}" tests ${test_index} properties
                        ${property_index} name)
                if (property_name STREQUAL "LABELS")
                    string(JSON label_count LENGTH
                            "${ctest_json}" tests ${test_index} properties
                            ${property_index} value)
                    if (label_count GREATER 0)
                        math(EXPR last_label "${label_count} - 1")
                        foreach (label_index RANGE 0 ${last_label})
                            string(JSON label GET
                                    "${ctest_json}" tests ${test_index} properties
                                    ${property_index} value ${label_index})
                            list(APPEND test_labels "${label}")
                        endforeach()
                    endif()
                endif()
            endforeach()
        endif()

        set(level_count 0)
        set(domain_count 0)
        foreach (label IN LISTS test_labels)
            list(FIND allowed_labels "${label}" allowed_index)
            if (allowed_index EQUAL -1)
                list(APPEND validation_errors
                        "${test_name}: undocumented label '${label}'")
            endif()
            list(FIND level_labels "${label}" level_index)
            if (NOT level_index EQUAL -1)
                math(EXPR level_count "${level_count} + 1")
            endif()
            list(FIND domain_labels "${label}" domain_index)
            if (NOT domain_index EQUAL -1)
                math(EXPR domain_count "${domain_count} + 1")
            endif()
        endforeach()

        if (NOT level_count EQUAL 1)
            list(APPEND validation_errors
                    "${test_name}: expected exactly one level label, found ${level_count}")
        endif()
        if (domain_count LESS 1)
            list(APPEND validation_errors
                    "${test_name}: expected at least one domain label")
        endif()
    endforeach()
endif()

if (validation_errors)
    list(JOIN validation_errors "\n  - " formatted_errors)
    message(FATAL_ERROR "Invalid CTest label metadata:\n  - ${formatted_errors}")
endif()

message(STATUS
        "Validated ${test_count} unique CTest registrations and their labels")
