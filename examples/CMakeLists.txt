# The examples, built with the rest of the project when
# STANCPP_BUILD_EXAMPLES is on, as it is by default at the top level:
#
#   cmake -S . -B build
#   cmake --build build
#
# Each example is one file, <name>.cpp, built into build/examples/<name>.
# It reads its data from <name>.json, which also runs <name>.stan in CmdStan.

file(GLOB example_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(source IN LISTS example_sources)
  get_filename_component(name "${source}" NAME_WE)
  add_executable(${name} "${source}")
  target_link_libraries(${name}
    PRIVATE stancpp_stan_math nlohmann_json::nlohmann_json stancpp::stancpp)
  # Each example finds its data file here unless given another path.
  target_compile_definitions(${name}
    PRIVATE EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  # With the tests on, ctest also runs each example, which must succeed.
  if(STANCPP_BUILD_TESTS)
    add_test(NAME examples.${name} COMMAND ${name})
  endif()
endforeach()
