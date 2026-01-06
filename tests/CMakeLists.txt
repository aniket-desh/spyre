# test configuration

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# test executable
add_executable(spyre_tests
    test_grid.cpp
    test_field.cpp
    test_transform.cpp
)

target_link_libraries(spyre_tests PRIVATE
    spyre
    Catch2::Catch2WithMain
)

# register tests with ctest
include(CTest)
include(Catch)
catch_discover_tests(spyre_tests)
