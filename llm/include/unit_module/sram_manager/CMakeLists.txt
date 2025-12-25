cmake_minimum_required(VERSION 3.10)

project(SramMemoryManagerProject)

include_directories(
    ${SYSTEMC_HOME}/include
    ../../../../llm/include
)

# Set C++ standard (optional, but good practice)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(sram_app main.cpp sram_manager.cpp)

# If your sram_manager.h is in a subdirectory like "include",
# you would add that directory to the include path:
# target_include_directories(sram_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# But since they are in the same directory for this example, it's not strictly needed.

# You can add compiler flags if needed
# target_compile_options(sram_app PRIVATE -Wall -Wextra -pedantic)

install(TARGETS sram_app DESTINATION bin)