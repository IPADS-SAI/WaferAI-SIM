#include "common/memory.h"
#include "utils/memory_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

#include <iostream>

using namespace std;

int AddrLabelTable::addRecord(const std::string &key) {

    for (int i = 0; i < table.size(); i++) {
        if (table[i] == key) {
            return i + 1;
        }
    }

    table.push_back(key);
    return table.size();
}

string AddrLabelTable::findRecord(int index) const {
    index -= 1;
    if (index >= 0 && index < table.size()) {
        return table[index];
    }
    return UNSET_LABEL;
}

void AddrLabelTable::clearAll() { table.clear(); }

void SramPosLocator::addPair(std::string &key, AddrPosKey value,
                             bool update_key) {
    visit += 1;
    value.record = visit;
    if (update_key) {
        data_map[key] = value;
    } else {
        AddrPosKey old_key;

        findPair(key, old_key);

        value.dram_addr = old_key.dram_addr;

        data_map[key] = value;
    }
}

void SramPosLocator::addPairByTile(std::string &key, AddrPosKey value,
                                   TaskCoreContext &context,
                                   u_int64_t &dram_time) {
    visit += 1;
    value.record = visit;

    // 需要检查原先是否有这个标签
    AddrPosKey old_key;
    int res = findPair(key, old_key);
    int old_size = 0;

    if (res == -1) {
        // 没找到，直接更新
        data_map[key] = value;
    } else {
        // 找到了，只更新大小
        old_size = old_key.size;
        old_key.size = value.size;
        old_key.record = value.record;
        data_map[key] = old_key;
    }

    LOG_DEBUG(MEMORY) << "Key " << key << " has old_size " << old_size
                      << " , spilled_size " << old_key.spill_size;

    // 检查所有的大小是否超过能够容纳的上限
    int used = 0;
    for (auto pair : data_map) {
        // valid = true 表示还没有被spill过
        if (pair.second.valid)
            used += pair.second.size;
        else
            used += pair.second.size - pair.second.spill_size;
    }

    // 放得下
    if (used <= max_sram_size) {
        LOG_DEBUG(MEMORY) << "Core " << cid << " has SRAM usage " << used
                          << " / " << max_sram_size;
        return;
    }

    LOG_INFO(MEMORY) << "Core " << cid << " need to spill SRAM";

    // 如果old_size不为0，则spill自己；如果为0，则spill除了自己以外record最大的数据。
    // 由于这里最多读取64*1024，所以固定spill 64*1024即可。
    if (used - max_sram_size > 64 * 1024)
        LOG_ERROR(memory.cpp)
            << "Core " << cid << " load tile bigger than 64*1024";

    int spill_limit = 64 * 1024;
    if (old_size > 0) {
        data_map[key].valid = false;
        data_map[key].spill_size += spill_limit;
        LOG_DEBUG(MEMORY) << "Core " << cid << " spill " << key << " to dram";
        LOG_DEBUG(MEMORY) << "Key " << key << " has spill_size "
                          << data_map[key].spill_size;
    } else {
        int max_record = -1;
        string max_label = "";
        for (auto pair : data_map) {
            if (pair.first == key)
                continue; // 不能spill自己
            if (!pair.second.valid &&
                pair.second.spill_size == pair.second.size)
                continue; // 已经全部spill到dram中去了

            if (pair.second.record > max_record) {
                max_record = pair.second.record;
                max_label = pair.first;
            }
        }

        if (max_record == -1) {
            LOG_ERROR(memory.cpp) << "SRAM have no more data to spill "
                                  << max_sram_size << "<" << used;
            return;
        }

        data_map[max_label].valid = false;
        data_map[max_label].spill_size += spill_limit;
        data_map[key].valid = true;

        LOG_DEBUG(MEMORY) << "Core " << cid << " spill " << max_label
                          << " to dram";
    }

    sram_spill_back_generic(context, spill_limit, 1024, dram_time);
}


bool SramPosLocator::validateTotalSize() const {
    int dataSizeSum = 0;
    for (const auto &pair : data_map) {
        if (pair.second.valid)
            dataSizeSum += pair.second.size - pair.second.spill_size;
    }

    int allocationSizeSum = 0;
    for (const auto &alloc : sram_manager_->allocations_) {
        AllocationID id = alloc.first;
        allocationSizeSum += sram_manager_->get_allocation_byte_capacity(id);
    }

    if (dataSizeSum != allocationSizeSum) {
        LOG_ERROR(memory.cpp) << "Total size validation failed, " << dataSizeSum
                              << " != " << allocationSizeSum;
        return false;
    }

    LOG_DEBUG(MEMORY) << "Total size validation passed: " << dataSizeSum
                      << " bytes.";
    return true;
}
void SramPosLocator::addPair(std::string &key, AddrPosKey value,
                             TaskCoreContext &context, u_int64_t &dram_time,
                             bool update_key) {
    // 先放入sram

    visit += 1;
    value.record = visit;
    if (update_key) {
        data_map[key] = value;
    } else {
        AddrPosKey old_key;

        findPair(key, old_key);

        value.dram_addr = old_key.dram_addr;

        data_map[key] = value;
    }

    // 检查所有的大小是否超过能够容纳的上限
    int used = 0;
    for (auto pair : data_map) {
        // valid = true 表示还没有被spill过
        if (pair.second.valid)
            used += pair.second.size;
        else
            used += pair.second.size - pair.second.spill_size;
    }

    // 放得下
    if (used <= max_sram_size) {
        LOG_DEBUG(MEMORY) << "Core " << cid << " has SRAM usage " << used;
        return;
    }

    LOG_INFO(MEMORY) << "Core " << cid << " need to spill SRAM";

    // 放不下，需要spill，查找里面record最小的成员（除了key）
    sc_time start_nbdram = sc_time_stamp();
    while (used > max_sram_size) {
        LOG_DEBUG(MEMORY) << "Core " << cid << " Sram check: used: " << used
                          << ", max sram size: " << max_sram_size;

        int min_record;
        if (SPEC_LOAD_STATIC == "layer")
            min_record = 1e9 + 3;
        else
            min_record = -1;

        string min_label = "";
        AllocationID sram_id = 0;
        for (auto pair : data_map) {
            if (pair.first == key)
                continue; // 不能spill自己
            if (!pair.second.valid &&
                pair.second.spill_size == pair.second.size)
                continue; // 已经全部spill到dram中去了

            if (SPEC_LOAD_STATIC == "layer") {
                if (pair.second.record < min_record) {
                    min_record = pair.second.record;
                    min_label = pair.first;
                    sram_id = pair.second.alloc_id;
                }
            } else {
                if (pair.second.record > min_record) {
                    min_record = pair.second.record;
                    min_label = pair.first;
                    sram_id = pair.second.alloc_id;
                }
            }
        }

        if (SPEC_LOAD_STATIC == "layer") {
            if (min_record == 1e9 + 3) {
                LOG_ERROR(memory.cpp) << "SRAM have no more data to spill "
                                      << max_sram_size << "<" << used;
                return;
            }
        } else {
            if (min_record == -1) {
                LOG_ERROR(memory.cpp) << "SRAM have no more data to spill "
                                      << max_sram_size << "<" << used;
                return;
            }
        }

        // 如果已经spill一部分了，则选择剩余能spill的大小
        int upper_spill_limit;
        if (data_map[min_label].valid) {
            upper_spill_limit = data_map[min_label].size;
        } else {
            upper_spill_limit =
                data_map[min_label].size - data_map[min_label].spill_size;
        }

        data_map[min_label].valid = false;

        int delta_space = used - max_sram_size;
        // 表示已经被放到dram中的数据大小
        // int spill_size =
        //     min(double(delta_space) * 1, (double)upper_spill_limit);
        int spill_size = upper_spill_limit;
        used -= spill_size;
        data_map[min_label].spill_size += spill_size;
        // data_map[min_label].size -= spill_size;
#if USE_SRAM_MANAGER == 1
        LOG_DEBUG(MEMORY) << "Core " << cid << " add pair to SRAM manager"
                          << key;
        sram_manager_->deallocate(sram_id);
        LOG_DEBUG(MEMORY) << "Core " << cid << " deallocate " << sram_id
                          << " from SRAM manager." << key;


        // spill 耗时
        // spill in nb_dcache utils
        LOG_DEBUG(MEMORY) << "Core " << cid << " spill to address "
                          << data_map[min_label].dram_addr;
        sram_spill_back_generic(context, spill_size,
                                data_map[min_label].dram_addr, dram_time);
#else
        std::string tail = min_label.substr(min_label.size() - 2);
        if (tail != "_w" && tail != "_b")
            sram_spill_back_generic(context, spill_size, 1024, dram_time);
#endif
    }
    sc_time end_nbdram = sc_time_stamp();
    u_int64_t nbdram_time = (end_nbdram - start_nbdram).to_seconds() * 1e9;
    LOG_DEBUG(MEMORY) << "Core " << cid << " finish spill SRAM, duration "
                      << nbdram_time;


    // 重排
    // 每次addPair后都需要重排sram_addr地址，保证最前面的一块是连续使用的，sram指向最前面空闲的
#if USE_SRAM_MANAGER
#else
    *(context.sram_addr) = rearrangeAll(context);
#endif
}

int SramPosLocator::findPair(std::string &key, int &result) {

    visit += 1;
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        it->second.record = visit;
        result = it->second.pos;
        return it->second.spill_size;
    }
    return -1;
}

void SramPosLocator::printAllKeys() {
    for (const auto &pair : data_map) {
        LOG_DEBUG(MEMORY_DEBUG) << "SRAM pos locator Key: " << pair.first;
    }
}
void SramPosLocator::printAllKeysWithAllocId() {
    LOG_DEBUG(MEMORY_DEBUG) << "SRAM pos locator Keys and Allocation IDs:";
    for (const auto &pair : data_map) {
        LOG_DEBUG(MEMORY_DEBUG) << "  Key: " << pair.first
                                << ", Alloc ID: " << pair.second.alloc_id;
    }
}
int SramPosLocator::findPair(std::string &key, AddrPosKey &result) {
    visit += 1;

    auto it = data_map.find(key);
    if (it != data_map.end()) {
        it->second.record = visit;
        result = it->second;
        return it->second.spill_size;
    }
    return -1;
}

int SramPosLocator::findKeySize(std::string &key) {


    auto it = data_map.find(key);
    if (it != data_map.end()) {
        return it->second.size;
    }
    return -1;
}


void SramPosLocator::updateKVPair(TaskCoreContext &context, std::string &key,
                                  uint64_t kv_daddr, int data_size_in_byte) {
#if USE_SRAM_MANAGER == 1
    visit += 1;

    AddrPosKey result;
    int spill_size = findPair(key, result);
    u_int64_t dram_time_tmp;

    if (spill_size == -1) {
        // 还未建立 KV sram block
        sram_write_append_generic(context, data_size_in_byte, dram_time_tmp,
                                  key, true, this, kv_daddr);
        return;


    } else if (spill_size > 0) {
        sram_first_write_generic(context, spill_size, kv_daddr, dram_time_tmp,
                                 nullptr, key, true, this);
        // KV sram block 之前被建立，但是被放回dram
    }

    spill_size = findPair(key, result);

    assert(spill_size >= 0);
    // assert(validateTotalSize());

    if (result.left_byte > data_size_in_byte) {
        result.spill_size = 0;
        result.left_byte -= data_size_in_byte;
        return;
    } else {
        int alignment =
            std::max(GetCoreHWConfig(cid)->sram_bitwidth, SRAM_BLOCK_SIZE * 8);
        int alignment_byte = alignment / 8;
        int tmp = 1;

        result.size += alignment_byte;
        while (tmp * alignment_byte < data_size_in_byte) {
            tmp = tmp + 1;
            result.size += alignment_byte;
        }

        result.left_byte =
            result.left_byte - data_size_in_byte + alignment_byte * tmp;

        addPair(key, result, context, dram_time_tmp, false);

        auto sram_manager_ = context.sram_manager_;
        sram_manager_->allocate_append(alignment_byte * tmp, result.alloc_id);
#if ASSERT == 1
        assert(validateTotalSize());
#endif
        return;
    }


#else
    assert(0);
#endif
}

void SramPosLocator::changePairName(std::string &old_key,
                                    std::string &new_key) {
    // 将旧标签名修改为新标签名
    AddrPosKey result;
    auto it = data_map.find(old_key);
    if (it != data_map.end()) {
        result = it->second;
        data_map.erase(it);
    }

    data_map[new_key] = result;
}

// 为sram中标签为key的数据块增加size的大小。如果该数据块还不存在，则创建一个。
void SramPosLocator::updatePair(std::string &key, int size,
                                TaskCoreContext &context,
                                u_int64_t &dram_time) {
    visit += 1;

    AddrPosKey result;
    int spill_size = findPair(key, result);

    if (spill_size == -1) {
        result.pos = *context.sram_addr;
        result.size = size;
    } else if (spill_size > 0) {
        // 需要先把所有内容取回
        sram_first_write_generic(context, spill_size, result.pos, dram_time,
                                 nullptr);
        result.spill_size = 0;
        result.size += size;

    } else {
        result.size += size;
    }

    addPair(key, result, context, dram_time);
}

void SramPosLocator::deletePair(std::string &key) {
    LOG_DEBUG(MEMORY_DEBUG)
        << "Core " << cid << " delete label " << key << " from SRAM";

    auto it = data_map.find(key);
    if (it != data_map.end()) {
#if USE_SRAM_MANAGER
        sram_manager_->deallocate(it->second.alloc_id); // 释放 SRAM
#endif
        data_map.erase(it);
    }
}

void SramPosLocator::clearAll() { data_map.clear(); }

int SramPosLocator::rearrangeAll(TaskCoreContext &context) {
    vector<pair<string, AddrPosKey>> temp_list;
    for (auto record : data_map)
        temp_list.push_back(record);

    clearAll();
    int pos = 0;
    for (auto record : temp_list) {
        auto size = record.second.size;
        auto spill_size = record.second.spill_size;

        int dma_read_count =
            spill_size * 8 /
            (int)(GetCoreHWConfig(cid)->sram_bitwidth * SRAM_BANKS);
        int byte_residue =
            spill_size * 8 -
            dma_read_count * (GetCoreHWConfig(cid)->sram_bitwidth * SRAM_BANKS);
        int single_read_count =
            CeilingDivision(byte_residue, GetCoreHWConfig(cid)->sram_bitwidth);

        int temp_pos = *(context.sram_addr);
        u_int64_t temp_addr = 0;
        LOG_DEBUG(MEMORY) << "Core " << cid << " SRAM key " << record.first
                          << " size " << size << " spilled size " << spill_size;

        addPair(record.first, record.second, context, temp_addr);

        if (temp_pos != *(context.sram_addr)) {
            LOG_ERROR(memory.cpp) << "Loop rearrange in SRAM spill";
        }

        pos += dma_read_count * SRAM_BANKS + single_read_count;
    }

    LOG_DEBUG(MEMORY) << "Core " << cid << " rearranged SRAM, new used size "
                      << pos;
    return pos;
}

// 以下为GpuPosLocator相关
void GpuPosLocator::addPair(const std::string &key, AddrPosKey &value) {
    value.pos = addr_top;
    data_map[key] = value;
    addr_top += value.size;

    // 对齐
    addr_top = CeilingDivision(addr_top, 64) * 64;
    LOG_DEBUG(GPU) << "Add pair: " << key << " pos: " << value.pos;
}

void GpuPosLocator::addPair(const std::string &key, AddrPosKey &value,
                            int size) {
    addr_top += size;
    data_map[key] = value;

    LOG_DEBUG(GPU) << "Add pair: " << key << " pos: " << value.pos;

    // 对齐
    addr_top = CeilingDivision(addr_top, 64) * 64;

    LOG_DEBUG(GPU) << "Update pair size: " << key << " pos: " << value.pos;
}

void GpuPosLocator::fetchPair(std::string &key, AddrPosKey &result) {
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        result = it->second;
        return;
    }

    addPair(key, result);
}

bool GpuPosLocator::findPair(std::string &key, int &result) {
    LOG_DEBUG(GPU) << "Try to find key: " << key;

    auto it = data_map.find(key);
    if (it != data_map.end()) {
        result = it->second.pos;
        return true;
    }

    LOG_WARN(GPU) << "Failed to find key: " << key;
    return false;
}

bool GpuPosLocator::findPair(std::string &key, AddrPosKey &result) {
    auto it = data_map.find(key);
    if (it != data_map.end()) {
        result = it->second;
        return true;
    }

    return false;
}

void GpuPosLocator::updatePair(std::string &key, int size) {
    AddrPosKey value = AddrPosKey(0, size);
    if (!findPair(key, value))
        addPair(key, value);
    else {
        value.size += size;
        addPair(key, value, size);
    }
}

void GpuPosLocator::deletePair(std::string &key) { data_map.erase(key); }

void GpuPosLocator::clearAll() { data_map.clear(); }