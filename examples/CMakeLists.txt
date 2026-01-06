# examples configuration

# c++ examples
add_executable(quick_test quick_test.cpp)
target_link_libraries(quick_test PRIVATE spyre)

add_executable(heat_example_cpp heat_example.cpp)
target_link_libraries(heat_example_cpp PRIVATE spyre)
