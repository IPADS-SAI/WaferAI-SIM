#!/bin/bash

# 脚本功能：
# 1. 运行 npusim 多组测试（3个json × 3个带宽）
# 2. 编译：cmake + make -j64
# 3. 再次运行同样的 npusim 测试（第二轮）
# 4. 完成

# 配置文件列表
# workload_configs=(
#     "pd_serving_gpt2_small12.json"
#     "pd_serving_gpt2_small13.json"
#     "pd_serving_gpt2_small14.json"
# )
workload_configs=(
    "pd_serving_gpt2_small12.json"
)

# 带宽列表
dram_bws=(256 512 1024)

# 公共参数
hardware_config="../llm/test/hardware_config/core_6x6.json"
gpu_inner="true"
use_gpu="true"

# 构建目录（假设你在 build 目录中运行此脚本）
build_dir="."
source_dir=".."  # npusim 源码根目录

# 函数：运行所有测试
run_tests() {
    echo "🚀 开始运行 npusim 测试..."

    for config in "${workload_configs[@]}"; do
        config_path="../llm/test/workload_config/gpu/$config"
        
        echo "=== 开始运行配置文件: $config ==="
        
        for bw in "${dram_bws[@]}"; do
            echo "⏳ 运行: ./npusim --workload-config $config --gpu_dram_bw $bw"
            
            ./npusim \
                --workload-config "$config_path" \
                --hardware-config "$hardware_config" \
                --gpu_inner "$gpu_inner" \
                --gpu_dram_bw "$bw" \
                --use_gpu "$use_gpu"
            
            if [ $? -ne 0 ]; then
                echo "❌ 错误：运行配置 $config 与带宽 $bw 失败。"
                exit 1
            fi
        done

        echo "✅ 完成配置文件: $config"
        echo
    done

    echo "🎉 所有测试运行完成！"
}

echo "🔧 开始重新编译：cmake + make -j64"
cmake -DBUILD_DEBUG_TARGETS=OFF -DL1CACHESIZE=4194304 -DL2CACHESIZE=15099494 ..
if [ $? -ne 0 ]; then
    echo "❌ cmake 配置失败"
    exit 1
fi

make -j64
if [ $? -ne 0 ]; then
    echo "❌ make 编译失败"
    exit 1
fi

echo "✅ 编译成功！"

# ============ 第一轮测试 ============
echo "🔁 第 1 轮测试"
run_tests

# ============ 重新编译 ============
echo "🔧 开始重新编译：cmake + make -j64"
cmake -DBUILD_DEBUG_TARGETS=OFF -DL1CACHESIZE=8388606 -DL2CACHESIZE=30198988 ..
if [ $? -ne 0 ]; then
    echo "❌ cmake 配置失败"
    exit 1
fi

make -j64
if [ $? -ne 0 ]; then
    echo "❌ make 编译失败"
    exit 1
fi

echo "✅ 编译成功！"

# ============ 第二轮测试 ============
echo "🔁 第 2 轮测试"
run_tests

echo "🎉🎉 两轮测试 + 重新编译已完成！"
# ============ 重新编译 ============
echo "🔧 开始重新编译：cmake + make -j64"
cmake -DBUILD_DEBUG_TARGETS=OFF -DL1CACHESIZE=16777212 -DL2CACHESIZE=60397976 ..
if [ $? -ne 0 ]; then
    echo "❌ cmake 配置失败"
    exit 1
fi

make -j64
if [ $? -ne 0 ]; then
    echo "❌ make 编译失败"
    exit 1
fi

echo "✅ 编译成功！"

# ============ 第二轮测试 ============
echo "🔁 第 3 轮测试"
run_tests

echo "🎉🎉 三轮测试 + 重新编译已完成！"