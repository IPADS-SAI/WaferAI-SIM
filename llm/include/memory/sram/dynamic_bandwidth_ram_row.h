#pragma once

#include "systemc.h"
#include <sysc/utils/sc_string.h>

#include "defs/const.h"
#include "memory/mem_if.h"
#include "memory/sram/arbiter_ram_bank.h"
#include "trace/Event_engine.h"

// start_address 是所有的地址位（包括末位对齐位置），不是index
// read_port_per_bank_ 每个 bank 能够同时读的数量
// read_port_num_ 每个 bank 总共有几个读端口

template<class T, unsigned BANK_NUM>
struct WriteRequest {
    uint64_t base_address;
    std::vector<T> data;
    sc_event done_event;     // 用于通知完成
    sc_time* elapsed_time;   // 返回耗时
    bool completed;          // 标记是否完成（可选）

    WriteRequest(uint64_t addr, const std::vector<T>& d, sc_time* et)
        : base_address(addr), data(d), elapsed_time(et), completed(false) {}
};

template <class T, unsigned BANK_NUM>
class DynamicBandwidthRamRow : public sc_channel,
                               public multiport_read_if<T, BANK_NUM>,
                               public ram_time_if<T> {
public:
    SC_HAS_PROCESS(DynamicBandwidthRamRow);
    DynamicBandwidthRamRow(sc_module_name name_, uint64_t start_address,
                           uint64_t depth_per_bank, uint64_t read_port_per_bank,
                           uint64_t write_port_per_bank, int read_port_num,
                           int write_port_num, int hb_read_port_num,
                           Event_engine *_e_engine)
        : write_request_fifo_(10),
          low_bw_start_address_(start_address),
          depth_per_bank_(depth_per_bank),
          read_port_per_bank_(read_port_per_bank),
          write_port_per_bank_(write_port_per_bank),
          read_port_num_(read_port_num),
          write_port_num_(write_port_num),
          hb_read_port_num_(hb_read_port_num) {
        low_bw_end_address_ = low_bw_start_address_ + BANK_NUM * depth_per_bank;

        assert(low_bw_start_address_ % BANK_NUM == 0);

        e_engine_ = _e_engine;

        for (auto i = 0; i < BANK_NUM; i++) {
            string ram_name_string = string("ram_bank_") + to_string(i);
            const char *ram_name = ram_name_string.data();
            ram_banks_.push_back(new ArbiterRamBank<T>(
                ram_name, 0, depth_per_bank_, read_port_per_bank_,
                write_port_per_bank_, e_engine_));
            addresses_.push_back(new sc_signal<uint64_t>());
            write_addresses_.push_back(new sc_signal<uint64_t>());
            data_from_banks_.push_back(new sc_signal<T>());
            data_to_banks_.push_back(new sc_signal<T>());
            ram_banks_[i]->address_read_in_parallel(*addresses_[i]);
            ram_banks_[i]->address_write_in_parallel(*write_addresses_[i]);
            ram_banks_[i]->data_read_in_parallel(*data_from_banks_[i]);
            ram_banks_[i]->data_write_in_parallel(*data_to_banks_[i]);
        }
        high_bw_read_semaphore_ =
            new sc_semaphore(1); // only support 1 request at a time
        high_bw_write_semaphore_ =
            new sc_semaphore(1); // only support 1 request at a time
        bound_read_port_num_ = 0;
        bound_write_port_num_ = 0;
        bound_high_bw_read_port_num_ = 0;
        SC_THREAD(process_write_requests);
    }

public:
    Event_engine *e_engine_;
    sc_fifo<WriteRequest<T, BANK_NUM>*> write_request_fifo_;

    void process_write_requests();  // 专用处理线程

    vector<ArbiterRamBank<T> *> ram_banks_;
    vector<sc_signal<uint64_t> *> addresses_;
    vector<sc_signal<uint64_t> *> write_addresses_;
    vector<sc_signal<T> *> data_from_banks_;
    vector<sc_signal<T> *> data_to_banks_;
    sc_semaphore *high_bw_read_semaphore_;
    sc_semaphore *high_bw_write_semaphore_;

    uint64_t low_bw_start_address_, low_bw_end_address_, depth_per_bank_;
    uint64_t high_bw_start_address_, high_bw_end_address_;
    uint64_t read_port_per_bank_, write_port_per_bank_;
    uint64_t bound_read_port_num_, bound_write_port_num_,
        bound_high_bw_read_port_num_;
    int read_port_num_, write_port_num_, hb_read_port_num_;

public:
    ~DynamicBandwidthRamRow();
    uint64_t get_bank_index(uint64_t address);
    uint64_t get_local_address(uint64_t address);

    void register_port(sc_port_base &port_, const char *if_typename_);
    transfer_status read(uint64_t address, T &data, sc_time &elapsed_time,
                         bool shadow = false);
    transfer_status multiport_read(uint64_t address_in_high_bitwidth,
                                   vector<T> &data, sc_time &elapsed_time);
    transfer_status multiport_write(uint64_t address_in_high_bitwidth,
                                    vector<T> &data, sc_time &elapsed_time);
    transfer_status write(uint64_t address, T &data, sc_time &elapsed_time,
                          bool force_write = 1, bool shadow = false);
    transfer_status clear(uint64_t address, T &data);
    bool reset();
    inline uint64_t start_address() const;
    inline uint64_t end_address() const;
};

template <class T, unsigned BANK_NUM>
inline DynamicBandwidthRamRow<T, BANK_NUM>::~DynamicBandwidthRamRow() {
    delete high_bw_read_semaphore_;
    delete high_bw_write_semaphore_;
    for (auto i = 0; i < BANK_NUM; i++) {
        delete ram_banks_[i];
        delete addresses_[i];
        delete data_from_banks_[i];
        delete write_addresses_[i];
        delete data_to_banks_[i];
    }
}

template <class T, unsigned BANK_NUM>
inline uint64_t
DynamicBandwidthRamRow<T, BANK_NUM>::get_bank_index(uint64_t address) {
    uint64_t column_idx = (address - low_bw_start_address_) % BANK_NUM;
    return column_idx;
}

template <class T, unsigned BANK_NUM>
inline uint64_t
DynamicBandwidthRamRow<T, BANK_NUM>::get_local_address(uint64_t address) {
    uint64_t row_relative_address =
        (address - low_bw_start_address_) % (depth_per_bank_ * BANK_NUM);
    return row_relative_address / (BANK_NUM);
}

template <class T, unsigned BANK_NUM>
void DynamicBandwidthRamRow<T, BANK_NUM>::process_write_requests() {
    while (true) {
        // 从 FIFO 读取请求（阻塞）
        WriteRequest<T, BANK_NUM>* request = write_request_fifo_.read();

        sc_time start_time = sc_time_stamp();
        sc_event_and_list event_all_done;

        uint64_t base_address = request->base_address;

        // 第一阶段：设置地址和数据，触发写
        {
            // 获取信号量（确保只有一个高带宽写在进行）
            if (high_bw_write_semaphore_->trywait() == -1) {
                high_bw_write_semaphore_->wait();
            }

            for (auto i = 0; i < BANK_NUM; i++) {
                uint64_t addr_offset = base_address + i;
                uint64_t bank_index = get_bank_index(addr_offset);
                uint64_t local_address = get_local_address(addr_offset);

                // 写入地址
                write_addresses_[bank_index]->write(local_address);

                // 写入数据
                data_to_banks_[i]->write(request->data[i]);

                // 收集完成事件
                event_all_done &= ram_banks_[bank_index]->write_done_event_;
            }

#if VERBOSE_TRACE == 1
            e_engine_->add_event(this->name(), "multiport_write", "B",
                                 Trace_event_util("write_processor"));
#endif

            // 触发并行写事件
            for (auto i = 0; i < BANK_NUM; i++) {
                uint64_t bank_index = get_bank_index(base_address + i);
                ram_banks_[bank_index]->write_in_parallel_event_.notify(SC_ZERO_TIME);
            }
        }

        // 等待所有 bank 写完成
        wait(event_all_done);

#if VERBOSE_TRACE == 1
        e_engine_->add_event(this->name(), "multiport_write", "E",
                             Trace_event_util("write_processor"));
#endif

        // 计算耗时
        *(request->elapsed_time) = sc_time_stamp() - start_time;

        // 释放信号量
        high_bw_write_semaphore_->post();

        // 通知原始线程完成
        request->done_event.notify(SC_ZERO_TIME);
    }
}

template <class T, unsigned BANK_NUM>
inline void
DynamicBandwidthRamRow<T, BANK_NUM>::register_port(sc_port_base &port_,
                                                   const char *if_typename_) {
    sc_module_name nm(if_typename_);
    if (nm == typeid(mem_read_if<T>).name()) {
        assert(bound_read_port_num_ < read_port_num_ || read_port_num_ < 0);
        bound_read_port_num_++;
    } else if (nm == typeid(mem_write_if<T>).name()) {
        assert(bound_write_port_num_ < write_port_num_ || write_port_num_ < 0);
        bound_write_port_num_++;
    } else if (nm == typeid(ram_if<T>).name()) {
        assert(bound_read_port_num_ < read_port_num_ || read_port_num_ < 0);
        assert(bound_write_port_num_ < write_port_num_ || write_port_num_ < 0);
        bound_read_port_num_++;
        bound_write_port_num_++;
    } else if (nm == typeid(multiport_read_if<T, BANK_NUM>).name()) {
        assert(bound_read_port_num_ < read_port_num_ || read_port_num_ < 0);
        assert(bound_high_bw_read_port_num_ < hb_read_port_num_ ||
               hb_read_port_num_ < 0); // only support one high-bw port for now
        bound_read_port_num_ += BANK_NUM;
        bound_high_bw_read_port_num_++;
    }
}
// address 是 index
template <class T, unsigned BANK_NUM>
inline transfer_status
DynamicBandwidthRamRow<T, BANK_NUM>::read(uint64_t address, T &data,
                                          sc_time &elapsed_time, bool shadow) {
    if (address < low_bw_start_address_ || address > low_bw_end_address_) {
        return TRANSFER_ERROR;
    }
    sc_time start_time = sc_time_stamp();
    uint64_t bank_index = get_bank_index(address);
    transfer_status temp_status =
        ram_banks_[bank_index]->read(get_local_address(address), data, shadow);
    elapsed_time = sc_time_stamp() - start_time;
    return temp_status;
}


template <class T, unsigned BANK_NUM>
inline transfer_status DynamicBandwidthRamRow<T, BANK_NUM>::multiport_write(
    uint64_t address_in_high_bitwidth, vector<T> &data, sc_time &elapsed_time) {
    
    assert(data.size() == BANK_NUM);

    // 创建请求对象（动态分配，由处理线程释放）
    auto request = new WriteRequest<T, BANK_NUM>(address_in_high_bitwidth, data, &elapsed_time);

    // 提交请求
    write_request_fifo_.write(request);

    // 等待完成
    wait(request->done_event);

    // 清理
    transfer_status status = TRANSFER_OK; // 可扩展为支持错误状态
    delete request;

    return status;
}

// template <class T, unsigned BANK_NUM>
// inline transfer_status DynamicBandwidthRamRow<T, BANK_NUM>::multiport_write(
//     uint64_t address_in_high_bitwidth, vector<T> &data, sc_time &elapsed_time) {
//     // 检查数据大小与银行数量是否匹配
//     assert(data.size() == BANK_NUM);
//     sc_time start_time = sc_time_stamp();
//     // 获取写信号量，确保同时只能有一个多端口写操作
//     if (high_bw_write_semaphore_->trywait() == -1) {
//         high_bw_write_semaphore_->wait(); // 阻塞直到信号量可用
//     }

//     uint64_t base_address = address_in_high_bitwidth;
//     sc_event_and_list event_all_done;

//     // 每个bank写入地址和数据
//     for (auto i = 0; i < BANK_NUM; i++) {
//         uint64_t bank_index = get_bank_index(base_address + i);
//         uint64_t local_address = get_local_address(base_address + i);
//         write_addresses_[bank_index]->write(local_address);

//         // 确保写入地址和数据合法
//         assert(bank_index < BANK_NUM);

//         event_all_done &= ram_banks_[bank_index]->write_done_event_;
//     }

//     // notify 每个bank 还是进行读
//     for (auto i = 0; i < BANK_NUM; i++) {
//         uint64_t bank_index = get_bank_index(base_address + i);
//         uint64_t local_address = get_local_address(base_address + i);
//         data_to_banks_[i]->write(data[i]);
//         ram_banks_[bank_index]->write_in_parallel_event_.notify(SC_ZERO_TIME);
//     }
// #if VERBOSE_TRACE == 1
//     e_engine_->add_event(this->name(), __func__, "B",
//                          Trace_event_util(sc_get_current_process_b()->name()));
// #endif
//     wait(event_all_done);
// #if VERBOSE_TRACE == 1
//     e_engine_->add_event(this->name(), __func__, "E",
//                          Trace_event_util(sc_get_current_process_b()->name()));
// #endif
//     // 释放信号量
//     high_bw_write_semaphore_->post();
//     elapsed_time = sc_time_stamp() - start_time;

//     return TRANSFER_OK;
// }


// 一次read 读出所有的bank
// 会调用 address_read_in_parallel
// 不能调用多个 read 这样读延迟也会增加
// address_in_high_bitwidth 也是地址的 index
// 一次性读出所有bank的数据，可以跨行
template <class T, unsigned BANK_NUM>
inline transfer_status DynamicBandwidthRamRow<T, BANK_NUM>::multiport_read(
    uint64_t address_in_high_bitwidth, vector<T> &data, sc_time &elapsed_time) {
    assert(data.size() == BANK_NUM);
    sc_time start_time = sc_time_stamp();
    if (high_bw_read_semaphore_->trywait() == -1) {
        high_bw_read_semaphore_
            ->wait(); // if the mutex is locked, this function will wait
    }

    uint64_t base_address = address_in_high_bitwidth; // * BANK_NUM;

    sc_event_and_list event_all_done;
    // 每个 bank 端口的读地址
    for (auto i = 0; i < BANK_NUM; i++) {
        uint64_t bank_index = get_bank_index(base_address + i);
        uint64_t local_address = get_local_address(base_address + i);
        addresses_[bank_index]->write(local_address);
        event_all_done &= ram_banks_[bank_index]->read_done_event_;
    }
    // notify 每个bank 还是进行读
    for (auto i = 0; i < BANK_NUM; i++) {
        uint64_t bank_index = get_bank_index(base_address + i);
        uint64_t local_address = get_local_address(base_address + i);
        ram_banks_[bank_index]->read_in_parallel_event_.notify(SC_ZERO_TIME);
    }
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), __func__, "B",
                         Trace_event_util(sc_get_current_process_b()->name()));
#endif
    wait(event_all_done);
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), __func__, "E",
                         Trace_event_util(sc_get_current_process_b()->name()));
#endif
    for (auto i = 0; i < BANK_NUM; i++) {
        data[i] = data_from_banks_[i]->read();
    }

    high_bw_read_semaphore_->post();
    elapsed_time = sc_time_stamp() - start_time;

    return TRANSFER_OK;
}

template <class T, unsigned BANK_NUM>
inline transfer_status
DynamicBandwidthRamRow<T, BANK_NUM>::write(uint64_t address, T &data,
                                           sc_time &elapsed_time,
                                           bool force_write, bool shadow) {
    if (address < low_bw_start_address_ || address > low_bw_end_address_) {
        return TRANSFER_ERROR;
    }
    sc_time start_time = sc_time_stamp();
    uint64_t bank_index = get_bank_index(address);
    transfer_status temp_status = ram_banks_[bank_index]->write(
        get_local_address(address), data, force_write, shadow);
    elapsed_time = sc_time_stamp() - start_time;
    return temp_status;
}

template <class T, unsigned BANK_NUM>
inline transfer_status
DynamicBandwidthRamRow<T, BANK_NUM>::clear(uint64_t address, T &data) {
    if (address < low_bw_start_address_ || address > low_bw_end_address_) {
        return TRANSFER_ERROR;
    }
    uint64_t bank_index = get_bank_index(address);
    transfer_status temp_status =
        ram_banks_[bank_index]->clear(get_local_address(address), data);
    return temp_status;
}

template <class T, unsigned BANK_NUM>
inline bool DynamicBandwidthRamRow<T, BANK_NUM>::reset() {
    assert(bound_read_port_num_ < read_port_num_ &&
           bound_write_port_num_ < write_port_num_ &&
           bound_high_bw_read_port_num_ < hb_read_port_num_);
    for (auto i = 0; i < BANK_NUM; i++) {
        ram_banks_[i]->reset();
    }
    return true;
}

template <class T, unsigned BANK_NUM>
inline uint64_t DynamicBandwidthRamRow<T, BANK_NUM>::start_address() const {
    return low_bw_start_address_;
}

template <class T, unsigned BANK_NUM>
inline uint64_t DynamicBandwidthRamRow<T, BANK_NUM>::end_address() const {
    return low_bw_end_address_;
}
