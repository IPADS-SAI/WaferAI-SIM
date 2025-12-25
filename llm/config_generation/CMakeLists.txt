cmake_minimum_required(VERSION 3.10)

project(ConfigGenerationProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
message(STATUS "SystemC Home: ${SYSTEMC_HOME}")
link_directories(${SYSTEMC_HOME}/lib-linux64)

include_directories(
    ${SYSTEMC_HOME}/include
    ../include
    # ../../DRAMSys/src/configuration
    # ../../DRAMSys/src/util
    # ../../DRAMSys/src/libdramsys
    
)
# add_subdirectory(./DRAMSys)
add_subdirectory(../../DRAMSys ${CMAKE_BINARY_DIR}/DRAMSys)
file(GLOB_RECURSE SRC_FILES ../src/*.cpp ../../llm/include/unit_module/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "../.*/unittest/.*")
# Remove 'test_dramsys.cpp' from the list of sources
list(FILTER SRC_FILES EXCLUDE REGEX "../.*/test_dramsys\\.cpp$")
list(FILTER SRC_FILES EXCLUDE REGEX "../.*/main\\.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX "../.*/build/.*")
message(STATUS "Source files: ${SRC_FILES}")



# Add the executable
add_executable(config_gen config_gen.cpp ${SRC_FILES})

target_link_libraries(config_gen
        PUBLIC
            systemc 
            cairo 
            sfml-graphics 
            sfml-system 
            sfml-window 
            m 
            pthread
            DRAMSys::libdramsys
            DRAMSys::util
            DRAMSys::config
    )