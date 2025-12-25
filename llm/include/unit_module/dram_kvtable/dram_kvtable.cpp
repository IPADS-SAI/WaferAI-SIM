#include "dram_kvtable.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cassert>
// 初始化静态常量
// const std::vector<int> DramKVTable::FIXED_VALUES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

DramKVTable::DramKVTable(uint64_t maxadddr, uint64_t kvcache_size, int n_entry)
     {

        for (int i = 1; i <= n_entry; ++i) {
            free_values.push_back(maxadddr - i * kvcache_size);
        }

        // for (auto &value : free_values) {
        //     std::cout << "Free value: " << value << std::endl;
        // }

     }

bool DramKVTable::add(const std::string& key) {
    if (free_values.empty()) {
        std::cerr << "No available value to assign.\n";
        assert(0);
        return false;
    }

    // 随机打乱 free_values，然后取第一个
    // std::shuffle(free_values.begin(), free_values.end(), rng);
    uint64_t selected_value = free_values.back();
    free_values.pop_back();

    map[key] = selected_value;
    return true;
}

bool DramKVTable::remove(const std::string& key) {
    auto it = map.find(key);
    if (it == map.end()) {
        return false;
    }

    uint64_t value = it->second;
    map.erase(it);
    free_values.push_back(value); // 把 value 放回空闲池
    return true;
}

std::optional<uint64_t> DramKVTable::get(const std::string& key) const {
    auto it = map.find(key);
    if (it != map.end()) {
        return it->second;
    }
    return {};
}

size_t DramKVTable::size() const {
    return map.size();
}

void DramKVTable::print() const {
    for (const auto& [k, v] : map) {
        std::cout << k << " => " << v << std::endl;
    }
}