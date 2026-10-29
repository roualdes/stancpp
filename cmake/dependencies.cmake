# What the tests and examples need beyond stancpp itself: Stan Math, with
# its TBB and SUNDIALS libraries, and nlohmann/json for reading data files.
#
# The library never includes this file, so a project that only uses
# stancpp fetches nothing and never mixes two Stan Math versions.
#
# Defines
#   stancpp_stan_math              Stan Math's headers, flags and libraries
#   nlohmann_json::nlohmann_json   nlohmann/json

include(FetchContent)
include(ProcessorCount)

# ------------------------------------------------------------
# Stan Math
# ------------------------------------------------------------

FetchContent_Declare(
  stan_math
  GIT_REPOSITORY https://github.com/stan-dev/math.git
  GIT_TAG        v5.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stan_math)

find_package(Threads REQUIRED)
find_program(MAKE_EXECUTABLE NAMES make gmake REQUIRED)

ProcessorCount(NPROC)
if(NOT NPROC OR NPROC EQUAL 0)
  set(NPROC 2)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(STAN_TBB_CXX_TYPE clang)
else()
  set(STAN_TBB_CXX_TYPE gcc)
endif()

# Stan Math builds these into its own lib/ folder; make skips them once built.
add_custom_target(stan_math_libs
  COMMAND "${MAKE_EXECUTABLE}" "-j${NPROC}"
          -f "${stan_math_SOURCE_DIR}/make/standalone" math-libs
          "CXX=${CMAKE_CXX_COMPILER}" "TBB_CXX_TYPE=${STAN_TBB_CXX_TYPE}"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Building Stan Math's TBB and SUNDIALS"
  VERBATIM
)

file(GLOB STAN_BOOST_DIR    LIST_DIRECTORIES true "${stan_math_SOURCE_DIR}/lib/boost_*")
file(GLOB STAN_SUNDIALS_DIR LIST_DIRECTORIES true "${stan_math_SOURCE_DIR}/lib/sundials_*")
file(GLOB STAN_TBB_DIR      LIST_DIRECTORIES true "${stan_math_SOURCE_DIR}/lib/tbb_*")
file(GLOB STAN_EIGEN_DIR    LIST_DIRECTORIES true "${stan_math_SOURCE_DIR}/lib/eigen_*")

add_library(stancpp_stan_math INTERFACE)
add_dependencies(stancpp_stan_math stan_math_libs)

target_compile_definitions(stancpp_stan_math
  INTERFACE _REENTRANT BOOST_DISABLE_ASSERTS EIGEN_NO_DEBUG STAN_THREADS
)

# As system headers, so the compiler reports no warnings from Stan Math and
# its libraries, such as TBB's uses of volatile that C++20 deprecates.
target_include_directories(stancpp_stan_math SYSTEM INTERFACE
  "${stan_math_SOURCE_DIR}"
  "${STAN_EIGEN_DIR}"
  "${STAN_BOOST_DIR}"
  "${STAN_SUNDIALS_DIR}/include"
  "${STAN_TBB_DIR}/include"
)

target_link_directories(stancpp_stan_math INTERFACE
  "${stan_math_SOURCE_DIR}/lib/tbb"
  "${STAN_SUNDIALS_DIR}/lib"
)

target_link_libraries(stancpp_stan_math INTERFACE
  tbb sundials_cvodes sundials_idas sundials_kinsol sundials_nvecserial
  Threads::Threads
)

# ------------------------------------------------------------
# nlohmann/json
# ------------------------------------------------------------

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)
