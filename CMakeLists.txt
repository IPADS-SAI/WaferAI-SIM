# 设置CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 定义项目名称
project(npusim_project LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 从环境变量读取 SYSTEMC_HOME
if(NOT DEFINED ENV{SYSTEMC_HOME})
    message(FATAL_ERROR "The environment variable SYSTEMC_HOME is not set. Please export SYSTEMC_HOME before running CMake.")
endif()

set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
message(STATUS "SystemC Home: ${SYSTEMC_HOME}")
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DBROAD_W=16")

# SystemC库路径
link_directories(${SYSTEMC_HOME}/lib-linux64)

# 包含目录
include_directories(
    ${SYSTEMC_HOME}/include
    ./llm/include
)


add_subdirectory(DRAMSys)

# 搜索所有.cpp文件
file(GLOB_RECURSE SRC_FILES ./llm/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX ".*/unittest/.*")
# Remove 'test_dramsys.cpp' from the list of sources
list(FILTER SRC_FILES EXCLUDE REGEX ".*/test_dramsys\\.cpp$")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/main\\.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/build/.*")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/test/.*")
list(FILTER SRC_FILES EXCLUDE REGEX ".*/config_generation/.*")


# # 定义生成的目标
message(STATUS "Source files: ${SRC_FILES}")

# find_package(Cairo REQUIRED)
# include_directories(${CAIRO_INCLUDE_DIRS})

option(BUILD_DEBUG_TARGETS "Build debug versions of executables" OFF)
# 设置 L1/L2 缓存大小（支持命令行传入，带默认值）
set(L1CACHESIZE ${L1CACHESIZE} CACHE STRING "L1 Cache Size in bytes")
set(L2CACHESIZE ${L2CACHESIZE} CACHE STRING "L2 Cache Size in bytes")

# 设置默认值（如果用户没传）
if(NOT L1CACHESIZE)
    set(L1CACHESIZE 4194304)
endif()

if(NOT L2CACHESIZE)
    set(L2CACHESIZE 15099494)
endif()

# 输出用于调试
message(STATUS "L1CACHESIZE: ${L1CACHESIZE}")
message(STATUS "L2CACHESIZE: ${L2CACHESIZE}")

function(add_test_executable target_name test_main_file)
    add_executable(${target_name} ${SRC_FILES} ${test_main_file})
    target_link_libraries(${target_name}
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
    target_compile_options(${target_name} PRIVATE 
        -Ofast 
        -Wno-unused-result 
        -Wno-ignored-pragmas 
        -Wno-unknown-attributes 
        -DBROAD_W=16
    )
    # 传递 L1CACHESIZE 和 L2CACHESIZE 到编译器
    target_compile_definitions(${target_name} PRIVATE
        L1CACHESIZE=${L1CACHESIZE}
        L2CACHESIZE=${L2CACHESIZE}
    )

    
    # 如果启用了调试版本构建选项，则添加调试版本
    if(BUILD_DEBUG_TARGETS)
        add_executable(${target_name}_debug ${SRC_FILES} ${test_main_file})
        target_link_libraries(${target_name}_debug
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
        target_compile_options(${target_name}_debug PRIVATE 
            -g 
            -O0 
            -Wno-unused-result 
            -Wno-ignored-pragmas 
            -Wno-unknown-attributes  
            -DBROAD_W=16
        )
        target_compile_definitions(${target_name}_debug PRIVATE
            L1CACHESIZE=${L1CACHESIZE}
            L2CACHESIZE=${L2CACHESIZE}
        )
    endif()
endfunction()

add_test_executable(npusim       "./llm/unittest/npusim.cpp")
# add_test_executable(load_config  "./llm/unittest/load_config.cpp")      
# add_test_executable(global_chip_test  "./llm/unittest/global_chip_test.cpp")

# function(add_debug_executable target_name test_main_file)
#     add_executable(${target_name}_debug ${SRC_FILES} ${test_main_file})
#     target_link_libraries(${target_name}_debug
#         PUBLIC
#             systemc 
#             cairo 
#             sfml-graphics 
#             sfml-system 
#             sfml-window 
#             m 
#             pthread
#             DRAMSys::libdramsys
#             DRAMSys::util
#             DRAMSys::config
#     )
#     target_compile_options(${target_name}_debug PRIVATE 
#         -g -O0 -Wall -Wextra -DBROAD_W=16 -DDEBUG
#     )
# endfunction()

# 添加Debug构建目标
# add_debug_executable(npusim       "./llm/unittest/npusim.cpp")
# add_debug_executable(load_config  "./llm/unittest/load_config.cpp")
# add_debug_executable(global_chip_test  "./llm/unittest/global_chip_test.cpp")

# 更新清理目标
# set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug;load_config;load_config_debug")


# add_executable(npusim ${SRC_FILES})
# target_link_libraries(npusim
#     PUBLIC
#     systemc cairo sfml-graphics sfml-system sfml-window m pthread
#         DRAMSys::libdramsys
#         DRAMSys::util
#         DRAMSys::config
# )
# target_compile_options(npusim PRIVATE -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16)

# # 调试版本
# add_executable(npusim_debug ${SRC_FILES})
# # set_target_properties(npusim_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
# # target_link_libraries(npusim_debug systemc sfml-graphics sfml-system sfml-window m)
# target_link_libraries(npusim_debug
#     PUBLIC
#     systemc cairo sfml-graphics sfml-system sfml-window m pthread
#         DRAMSys::libdramsys
#         DRAMSys::util
#         DRAMSys::config
# )
# target_compile_options(npusim_debug PRIVATE -g -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -DBROAD_W=16)

# 清理目标的功能
# set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug")
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "npusim;npusim_debug;load_config;load_config_debug")# add_library(systemc INTERFACE)

# add_library(SystemC::systemc ALIAS systemc)

