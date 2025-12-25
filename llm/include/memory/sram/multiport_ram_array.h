#pragma once

#include "systemc.h"
#include <sysc/utils/sc_string.h>

#include "defs/const.h"
#include "mem_if.h"
#include "ram.h"
#include "trace/Event_engine.h"


// 多 bank 每个 bank 可以连接多个读和写端口，每个 bank 可以同时支持 1
// 个或多个读或写请求 每个 bank 有独立的 sc_semaphore
/// @brief A model of sram bank array with multiple read and write port
/// @tparam T Data type in a sram cell
/// @tparam READ_PORT_NUM Number of read ports that should be connected
/// @tparam WRITE_PORT_NUM Number of write ports that should be connected
template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
class MultiportRamArray : public sc_channel, public ram_if<T> {
public:
    // read_port_per_bank_ 表示可以有几个端口同时都，
    // READ_PORT_NUM 表示有几个读端口
    // start_address 是 index
    MultiportRamArray(sc_module_name name_, uint64_t start_address,
                      uint64_t bank_num, uint64_t depth_per_bank,
                      uint64_t read_port_per_bank, uint64_t write_port_per_bank,
                      Event_engine *_e_engine)
        : start_address_(start_address),
          bank_num_(bank_num),
          depth_per_bank_(depth_per_bank),
          read_port_per_bank_(read_port_per_bank),
          write_port_per_bank_(write_port_per_bank) {
        end_address_ = start_address_ + bank_num * depth_per_bank;
        assert(end_address_ >= start_address_);

        e_engine_ = _e_engine;

        uint64_t start_address_of_bank = start_address;
        for (auto i = 0; i < bank_num; i++) {
            string ram_name_string = string("ram_bank_") + to_string(i);
            const char *ram_name = ram_name_string.data();
            ram_banks_.push_back(
                new Ram<T>(ram_name, start_address_of_bank,
                           start_address_of_bank + depth_per_bank, e_engine_));
            write_semaphores_.push_back(new sc_semaphore(write_port_per_bank_));
            read_semaphores_.push_back(new sc_semaphore(read_port_per_bank_));
            start_address_of_bank += depth_per_bank;
        }

        bound_read_port_num_ = 0;
        bound_write_port_num_ = 0;
    }

    ~MultiportRamArray() {
        for (auto i = 0; i < bank_num_; i++) {
            delete ram_banks_[i];
            delete read_semaphores_[i];
            delete write_semaphores_[i];
        }
    }
    void register_port(sc_port_base &port_, const char *if_typename_);
    uint64_t get_bank_index(uint64_t address);
    transfer_status read(uint64_t address, T &data, bool shadow = false);
    transfer_status write(uint64_t address, T &data, bool force_write = 1,
                          bool shadow = false);
    transfer_status clear(uint64_t address, T &data);
    bool reset();
    inline uint64_t start_address() const;
    inline uint64_t end_address() const;

public:
    Event_engine *e_engine_;

    vector<Ram<T> *> ram_banks_;
    vector<sc_semaphore *> read_semaphores_;
    vector<sc_semaphore *> write_semaphores_;

    uint64_t start_address_, end_address_, depth_per_bank_;
    uint64_t bank_num_, read_port_per_bank_, write_port_per_bank_;
    uint64_t bound_read_port_num_, bound_write_port_num_;
};

/// @brief Overload of a SystemC function in sc_interface. Count the port num
/// connected to this ram array.
// 替代默认接口注册函数，读写端口的数量要跟 READ_PORT_NUM 和 WRITE_PORT_NUM 相同
template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline void MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::register_port(
    sc_port_base &port_, const char *if_typename_) {
    sc_module_name nm(if_typename_);
    if (nm == typeid(mem_read_if<T>).name()) {
        assert(bound_read_port_num_ < READ_PORT_NUM);
        bound_read_port_num_++;
    } else if (nm == typeid(mem_write_if<T>).name()) {
        assert(bound_write_port_num_ < WRITE_PORT_NUM);
        bound_write_port_num_++;
    } else if (nm == typeid(ram_if<T>).name()) {
        assert(bound_read_port_num_ < READ_PORT_NUM);
        assert(bound_write_port_num_ < WRITE_PORT_NUM);
        bound_read_port_num_++;
        bound_write_port_num_++;
    }
}

/// @brief Get the bank index of the address
template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline uint64_t
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::get_bank_index(
    uint64_t address) {
    return address / depth_per_bank_;
}

/// @brief
template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline transfer_status
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::read(uint64_t address,
                                                          T &data,
                                                          bool shadow) {
    if (address < start_address_ || address > end_address_) {
        return TRANSFER_ERROR;
    }
    uint64_t bank_index = get_bank_index(address);

    e_engine_->add_event(
        this->name(),
        string("bank") + string("[") + to_string(bank_index) + string("]") +
            string(".read"),
        "C",
        Trace_event_util((float)read_port_per_bank_ -
                         read_semaphores_[bank_index]->get_value()));

    if (read_semaphores_[bank_index]->trywait() == -1) {
        read_semaphores_[bank_index]
            ->wait(); // if the mutex is locked, this function will wait
    }

    transfer_status temp_status = ram_banks_[bank_index]->read(address, data);
    read_semaphores_[bank_index]->post();
    return temp_status;
}


/// @brief
template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline transfer_status
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::write(uint64_t address,
                                                           T &data,
                                                           bool force_write,
                                                           bool shadow) {
    if (address < start_address_ || address > end_address_) {
        return TRANSFER_ERROR;
    }
    uint64_t bank_index = get_bank_index(address);

    e_engine_->add_event(
        this->name(),
        string("bank") + string("[") + to_string(bank_index) + string("]") +
            string(".write"),
        "C",
        Trace_event_util((float)write_port_per_bank_ -
                         write_semaphores_[bank_index]->get_value()));

    if (write_semaphores_[bank_index]->trywait() == -1) {
        write_semaphores_[bank_index]
            ->wait(); // if the mutex is locked, this function will wait
    }
    transfer_status temp_status =
        ram_banks_[bank_index]->write(address, data, force_write);
    write_semaphores_[bank_index]->post();
    return temp_status;
}

template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline transfer_status
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::clear(uint64_t address,
                                                           T &data) {
    if (address < start_address_ || address > end_address_) {
        return TRANSFER_ERROR;
    }
    uint64_t bank_index = get_bank_index(address);
    return ram_banks_[bank_index]->clear(address, data);
}

template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline bool MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::reset() {
    assert(bound_read_port_num_ == READ_PORT_NUM &&
           bound_write_port_num_ == WRITE_PORT_NUM);
    for (auto i = 0; i < bank_num_; i++) {
        ram_banks_[i]->reset();
    }
    return true;
}

template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline uint64_t
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::start_address() const {
    return start_address_;
}

template <class T, unsigned READ_PORT_NUM, unsigned WRITE_PORT_NUM>
inline uint64_t
MultiportRamArray<T, READ_PORT_NUM, WRITE_PORT_NUM>::end_address() const {
    return end_address_;
}
