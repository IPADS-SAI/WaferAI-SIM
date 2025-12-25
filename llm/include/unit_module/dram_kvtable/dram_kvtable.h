#pragma once


#include <unordered_map>
#include <string>
#include <vector>
#include <random>
#include <optional>
#include <stdexcept>

class DramKVTable {
private:


    // 当前可用的 value 集合
    std::vector<uint64_t> free_values;

    // key -> value 映射
    std::unordered_map<std::string, uint64_t> map;

public:
    DramKVTable(uint64_t maxadddr, uint64_t kvcache_size, int n_entry);

    // 添加 key，自动分配一个空闲 value
    bool add(const std::string& key);

    // 删除 key，释放其 value
    bool remove(const std::string& key);

    // 查找 key 对应的 value
    std::optional<uint64_t> get(const std::string& key) const;

    // 获取当前已使用的条目数
    size_t size() const;

    // 打印所有映射
    void print() const;
};
