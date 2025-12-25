cmake_minimum_required(VERSION 3.14)
project(DramKVTable)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR})

# 添加可执行文件
add_executable(dramkvtable main.cpp dram_kvtable.cpp)

# 如果你想以后支持库分离，也可以这样写：
# add_library(FixedStringMap STATIC include/FixedStringMap.hpp)
# target_link_libraries(fixed_string_map_app FixedStringMap)