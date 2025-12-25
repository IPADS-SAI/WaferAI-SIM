#pragma once

#include "systemc.h"
#include <sysc/utils/sc_string.h>

#include "memory/mem_if.h"
#include "memory/ram.h"
#include "trace/Event_engine.h"
// DynamicBandwidthRamRow 中 被使用
//  start_address 是 index
//  read_port_per_bank_ 每个 bank 能够同时读的数量
template <class T> class ArbiterRamBank : public sc_channel, public ram_if<T> {
public:
    // 相较于 multiport_ram_array 只有一个 bank
    SC_HAS_PROCESS(ArbiterRamBank);
    ArbiterRamBank(sc_module_name name_, uint64_t start_address,
                   uint64_t depth_per_bank, uint64_t read_port_per_bank,
                   uint64_t write_port_per_bank, Event_engine *_e_engine)
        : start_address_(start_address),
          depth_per_bank_(depth_per_bank),
          read_port_per_bank_(read_port_per_bank),
          write_port_per_bank_(write_port_per_bank) {
        end_address_ = start_address_ + depth_per_bank_;
        assert(end_address_ >= start_address_);

        e_engine_ = _e_engine;
        write_shadow = true;

        uint64_t start_address_of_bank = start_address;

        std::string modified_name = this->name();
        std::replace(modified_name.begin(), modified_name.end(), '.', '_');

        ram_bank_ = new Ram<T>(sc_core::sc_module_name(modified_name.c_str()),
                               start_address_, end_address_, e_engine_);
        write_semaphore_ = new sc_semaphore(write_port_per_bank_);
        read_semaphore_ = new sc_semaphore(read_port_per_bank_);

        SC_THREAD(read_in_parallel);
        sensitive << read_in_parallel_event_;
        dont_initialize();

        SC_THREAD(write_in_parallel);
        sensitive << write_in_parallel_event_;
        dont_initialize();
    }

public:
    ~ArbiterRamBank() {
        delete ram_bank_;
        delete read_semaphore_;
        delete write_semaphore_;
    }
    void read_in_parallel();
    void write_in_parallel();
    transfer_status read(uint64_t address, T &data, bool shadow = false);
    transfer_status write(uint64_t address, T &data, bool force_write = 1,
                          bool shadow = false);
    transfer_status clear(uint64_t address, T &data);
    bool reset();
    inline uint64_t start_address() const;
    inline uint64_t end_address() const;

public:
    Event_engine *e_engine_;
    bool write_shadow;

    Ram<T> *ram_bank_;
    sc_semaphore *read_semaphore_;
    sc_semaphore *write_semaphore_;

    sc_event_queue read_in_parallel_event_;
    sc_event_queue write_in_parallel_event_;
    sc_event read_done_event_;
    sc_event write_done_event_;

    sc_in<uint64_t> address_read_in_parallel;
    sc_out<T> data_read_in_parallel;
    sc_out<uint64_t> address_write_in_parallel;
    sc_in<T> data_write_in_parallel;

    uint64_t start_address_, end_address_, depth_per_bank_;
    uint64_t read_port_per_bank_, write_port_per_bank_;
};

template <class T> inline void ArbiterRamBank<T>::write_in_parallel() {
    while (true) {
        if (write_semaphore_->trywait() == -1) {
            write_semaphore_->wait(); // 如果信号量被锁定，这里会阻塞等待
        }

        T data_temp = data_write_in_parallel.read(); // 从输入端口读取数据
        uint64_t address_temp =
            address_write_in_parallel.read(); // 从输入端口读取地址
#if VERBOSE_TRACE == 1
        e_engine_->add_event(this->name(), "Write simultaneously", "C",
                             Trace_event_util((float)write_port_per_bank_ -
                                              write_semaphore_->get_value()));
#endif
        // 执行写操作，确保地址在范围内
        assert(ram_bank_->write(address_temp, data_temp, write_shadow) ==
               TRANSFER_OK);

        write_semaphore_->post(); // 释放信号量
#if VERBOSE_TRACE == 1
        e_engine_->add_event(this->name(), "Write simultaneously", "C",
                             Trace_event_util((float)write_port_per_bank_ -
                                              write_semaphore_->get_value()));
#endif
        write_done_event_.notify();
        wait(); // 等待新的写事件到来
    }
}

template <class T> inline void ArbiterRamBank<T>::read_in_parallel() {
    while (true) {
        if (read_semaphore_->trywait() == -1) {
            read_semaphore_
                ->wait(); // if the mutex is locked, this function will wait
        }
        T data_temp;
#if VERBOSE_TRACE == 1
        e_engine_->add_event(this->name(), "Read simultaneously", "C",
                             Trace_event_util((float)read_port_per_bank_ -
                                              read_semaphore_->get_value()));
#endif
        assert(ram_bank_->read(address_read_in_parallel.read(), data_temp) ==
               TRANSFER_OK);
        data_read_in_parallel.write(data_temp);
        read_semaphore_->post();
#if VERBOSE_TRACE == 1
        e_engine_->add_event(this->name(), "Read simultaneously", "C",
                             Trace_event_util((float)read_port_per_bank_ -
                                              read_semaphore_->get_value()));
#endif
        read_done_event_.notify();
        wait();
    }
}

template <class T>
inline transfer_status ArbiterRamBank<T>::read(uint64_t address, T &data,
                                               bool shadow) {
    if (address < start_address_ || address > end_address_) {
        return TRANSFER_ERROR;
    }

    if (read_semaphore_->trywait() == -1) {
        read_semaphore_
            ->wait(); // if the mutex is locked, this function will wait
    }
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), "Read simultaneously", "C",
                         Trace_event_util((float)read_port_per_bank_ -
                                          read_semaphore_->get_value()));
#endif
    transfer_status temp_status = ram_bank_->read(address, data, shadow);
    read_semaphore_->post();
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), "Read simultaneously", "C",
                         Trace_event_util((float)read_port_per_bank_ -
                                          read_semaphore_->get_value()));
#endif
    return temp_status;
}

template <class T>
inline transfer_status ArbiterRamBank<T>::write(uint64_t address, T &data,
                                                bool force_write, bool shadow) {
    if (address < start_address_ || address > end_address_) {
        return TRANSFER_ERROR;
    }
    if (write_semaphore_->trywait() == -1) {
        write_semaphore_
            ->wait(); // if the mutex is locked, this function will wait
    }
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), "Write simultaneously", "C",
                         Trace_event_util((float)write_port_per_bank_ -
                                          write_semaphore_->get_value()));
#endif
    transfer_status temp_status =
        ram_bank_->write(address, data, force_write, shadow);
    write_semaphore_->post();
#if VERBOSE_TRACE == 1
    e_engine_->add_event(this->name(), "Write simultaneously", "C",
                         Trace_event_util((float)write_port_per_bank_ -
                                          write_semaphore_->get_value()));
#endif
    return temp_status;
}

template <class T>
inline transfer_status ArbiterRamBank<T>::clear(uint64_t address, T &data) {
    return ram_bank_->clear(address, data);
}

template <class T> inline bool ArbiterRamBank<T>::reset() {
    return ram_bank_->reset();
}

template <class T> inline uint64_t ArbiterRamBank<T>::start_address() const {
    return start_address_;
}

template <class T> inline uint64_t ArbiterRamBank<T>::end_address() const {
    return end_address_;
}
