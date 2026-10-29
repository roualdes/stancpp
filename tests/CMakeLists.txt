# The tests, built with the rest of the project when STANCPP_BUILD_TESTS is
# on, as it is by default at the top level. Run them all with
#
#   ctest --test-dir build --output-on-failure
#
# Each test is one file, <name>_test.cpp, built into build/tests/<name>_test
# and added with stancpp_add_test(<name>). The models compare themselves with
# CmdStan values stored in reference/; reference/update.sh rebuilds those.

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Build <name>_test.cpp into a test executable and register each of its
# tests with ctest.
function(stancpp_add_test name)
  add_executable(${name}_test ${name}_test.cpp)
  target_include_directories(${name}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name}_test
    PRIVATE
      stancpp_stan_math
      nlohmann_json::nlohmann_json
      stancpp::stancpp
      GTest::gtest_main
  )
  # The tests find their reference files here.
  target_compile_definitions(${name}_test
    PRIVATE TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  gtest_discover_tests(${name}_test)
endfunction()

stancpp_add_test(spec)
stancpp_add_test(scope)
stancpp_add_test(data)
stancpp_add_test(model)
stancpp_add_test(decomposition)
stancpp_add_test(mvn)
