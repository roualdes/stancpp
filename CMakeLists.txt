cmake_minimum_required(VERSION 3.24)

project(stancpp
  VERSION 0.1.0
  DESCRIPTION "Stan models in C++"
  LANGUAGES CXX
)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
endif()

# ------------------------------------------------------------
# The library
# ------------------------------------------------------------
#
# Header-only and dependency-light on purpose: it needs Stan Math and Eigen
# at *compile* time, but it does not fetch them.  The consumer already has
# them, and fetching a second copy would risk two different Stan Math
# versions in one translation unit -- exactly the failure this project ran
# into with the simplex transform.
#
# So: link stancpp::stancpp alongside whatever target already provides Stan
# Math, e.g.
#
#     target_link_libraries(my_model INTERFACE my_stan_math stancpp::stancpp)

add_library(stancpp INTERFACE)
add_library(stancpp::stancpp ALIAS stancpp)

target_compile_features(stancpp INTERFACE cxx_std_20)

target_include_directories(stancpp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
#
# The tests and examples are built by default only when this is the
# top-level project. They fetch Stan Math, which the library itself never
# does.

option(STANCPP_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(STANCPP_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(STANCPP_INSTALL "Generate install rules for stancpp" OFF)

if(STANCPP_BUILD_TESTS OR STANCPP_BUILD_EXAMPLES)
  include(cmake/dependencies.cmake)
endif()

# Before the examples, which become tests too when the tests are on.
if(STANCPP_BUILD_TESTS)
  enable_testing()
endif()

if(STANCPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(STANCPP_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------
# Install (opt in)
# ------------------------------------------------------------

if(STANCPP_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(DIRECTORY include/stancpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(TARGETS stancpp EXPORT stancppTargets)

  install(EXPORT stancppTargets
    FILE stancppTargets.cmake
    NAMESPACE stancpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stancpp
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/stancppConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stancppConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stancpp
  )
endif()
