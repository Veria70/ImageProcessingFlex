file(GLOB IPF_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)

add_executable(ipf_tests ${IPF_TEST_SOURCES})
target_link_libraries(ipf_tests PRIVATE ipf::ipf)
target_compile_definitions(ipf_tests PRIVATE IPF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

if(MSVC)
    target_compile_options(ipf_tests PRIVATE /W4 /WX /permissive- /utf-8 /bigobj)
else()
    target_compile_options(ipf_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()

add_test(NAME ipf_tests COMMAND ipf_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Cross-validation tool driven by tests/crossval/run_crossval.py (needs Python + numpy/scipy/Pillow).
add_executable(ipf_crossval crossval/crossval.cpp)
target_link_libraries(ipf_crossval PRIVATE ipf::ipf)
if(MSVC)
    target_compile_options(ipf_crossval PRIVATE /W4 /WX /permissive- /utf-8 /bigobj)
endif()

# Every public header must compile on its own (no hidden include-order dependencies).
file(GLOB_RECURSE IPF_PUBLIC_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/ipf/*.hpp)
set(IPF_HEADER_CHECK_SOURCES "")
foreach(hdr ${IPF_PUBLIC_HEADERS})
    string(MAKE_C_IDENTIFIER ${hdr} id)
    set(src ${CMAKE_CURRENT_BINARY_DIR}/header_check/${id}.cpp)
    file(WRITE ${src} "#include <${hdr}>\nint ipf_header_check_${id}() { return 0; }\n")
    list(APPEND IPF_HEADER_CHECK_SOURCES ${src})
endforeach()
add_library(ipf_header_check OBJECT ${IPF_HEADER_CHECK_SOURCES})
target_link_libraries(ipf_header_check PRIVATE ipf::ipf)
if(MSVC)
    target_compile_options(ipf_header_check PRIVATE /W4 /WX /permissive-)
endif()

option(IPF_SANITIZE "Build the tests with AddressSanitizer" OFF)
if(IPF_SANITIZE)
    if(MSVC)
        target_compile_options(ipf_tests PRIVATE /fsanitize=address /Zi)
        target_link_options(ipf_tests PRIVATE /INCREMENTAL:NO)
    else()
        target_compile_options(ipf_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(ipf_tests PRIVATE -fsanitize=address,undefined)
    endif()
endif()

# Optional: cross-validation against Pillow/numpy/scipy (requires Python 3 with those packages).
option(IPF_CROSSVAL "Register the Python cross-validation suite with CTest" OFF)
if(IPF_CROSSVAL)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME ipf_crossval
             COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/crossval/run_crossval.py $<TARGET_FILE:ipf_crossval>)
endif()
