cmake_minimum_required(VERSION 3.18)
project(spyre
    VERSION 0.1.0
    DESCRIPTION "fast spherical pde solver library"
    LANGUAGES CXX
)

# c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# options
option(SPYRE_BUILD_TESTS "build tests" ON)
option(SPYRE_BUILD_PYTHON "build python bindings" ON)
option(SPYRE_BUILD_EXAMPLES "build examples" ON)
option(SPYRE_USE_OPENMP "enable openmp parallelization" ON)

# find dependencies
find_package(Eigen3 REQUIRED)

# find shtns (optional)
option(SPYRE_USE_SHTNS "use shtns library for transforms" ON)
find_library(SHTNS_LIBRARY shtns HINTS /usr/local/lib /opt/homebrew/lib)
find_path(SHTNS_INCLUDE_DIR shtns.h HINTS /usr/local/include /opt/homebrew/include)
if(NOT SHTNS_LIBRARY OR NOT SHTNS_INCLUDE_DIR)
    message(WARNING "shtns not found - using fallback implementation (slower)")
    message(STATUS "for best performance, install shtns from https://bitbucket.org/nschaeff/shtns")
    set(SPYRE_USE_SHTNS OFF)
endif()

# find fftw3 (required for shtns, optional otherwise)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 fftw3)
endif()
if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIBRARIES fftw3 HINTS /usr/local/lib /opt/homebrew/lib)
    find_path(FFTW3_INCLUDE_DIRS fftw3.h HINTS /usr/local/include /opt/homebrew/include)
endif()

# openmp
if(SPYRE_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "openmp found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# main library
add_library(spyre
    src/core/field.cpp
    src/grid/gauss_legendre.cpp
    src/grid/equiangular.cpp
    src/transform/spherical_harmonics.cpp
    src/solver/heat.cpp
)

target_include_directories(spyre
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(spyre
    PUBLIC
        Eigen3::Eigen
)

# add shtns if available
if(SPYRE_USE_SHTNS)
    find_library(FFTW3_LIB fftw3 HINTS /opt/homebrew/lib /usr/local/lib REQUIRED)
    target_include_directories(spyre PRIVATE ${SHTNS_INCLUDE_DIR} ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(spyre PUBLIC ${SHTNS_LIBRARY} ${FFTW3_LIB})
    target_compile_definitions(spyre PRIVATE SPYRE_USE_SHTNS)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(spyre PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(spyre PRIVATE SPYRE_USE_OPENMP)
endif()

# python bindings
if(SPYRE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG QUIET)

    if(Python3_FOUND AND pybind11_FOUND)
        pybind11_add_module(_spyre_core python/src/bindings.cpp)
        target_link_libraries(_spyre_core PRIVATE spyre)
        install(TARGETS _spyre_core DESTINATION python/spyre)
        message(STATUS "python bindings enabled")
    else()
        message(WARNING "pybind11 not found - python bindings disabled")
        message(STATUS "install with: pip install pybind11 && pip install pybind11[global]")
    endif()
endif()

# tests
if(SPYRE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# examples
if(SPYRE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# install
install(TARGETS spyre
    EXPORT spyre-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/spyre DESTINATION include)
