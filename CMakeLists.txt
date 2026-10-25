cmake_minimum_required(VERSION 3.14...3.31)
project(ImageProcessingFlex VERSION 1.1.0
        DESCRIPTION "Header-only, dependency-free C++17 image processing library"
        HOMEPAGE_URL "https://github.com/Veria70/ImageProcessingFlex"
        LANGUAGES CXX)

# Header-only library target: link against ipf::ipf and include <ipf/ipf.hpp>.
add_library(ipf INTERFACE)
add_library(ipf::ipf ALIAS ipf)
target_include_directories(ipf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(ipf INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(IPF_IS_TOP_LEVEL ON)
else()
    set(IPF_IS_TOP_LEVEL OFF)
endif()

option(IPF_BUILD_TESTS "Build the ipf unit tests" ${IPF_IS_TOP_LEVEL})
option(IPF_BUILD_EXAMPLES "Build the ipf examples" ${IPF_IS_TOP_LEVEL})

if(IPF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(IPF_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
install(TARGETS ipf EXPORT ipfTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/ipf)
install(EXPORT ipfTargets NAMESPACE ipf:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ipf)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ipfConfig.cmake
    "include(\${CMAKE_CURRENT_LIST_DIR}/ipfTargets.cmake)\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ipfConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ipf)
