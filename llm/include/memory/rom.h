#pragma once
#include "systemc.h"

#include "defs/const.h"
#include "mem_if.h"
#include "trace/Event_engine.h"


template <class T> class rom : public sc_channel, public rom_if<T> {
private:
    T *mem;
    uint64_t start_address_;
    uint64_t end_address_;

public:
    Event_engine *e_engine;

public:
    rom(sc_module_name name_, uint64_t start_address, uint64_t end_address,
        T *&data, Event_engine *_e_engine)
        : start_address_(start_address), end_address_(end_address) {
        // sc_assert(end_address_ >= m_start_addresss);
        e_engine = _e_engine;
        mem = new T[end_address_ - start_address_];

        for (uint64_t i = start_address; i < end_address; i++) {
            mem[i - start_address] = data[i - start_address];
        }
    }
    ~rom() {
        if (mem) {
            delete[] mem;
        }
    }

    virtual void register_port(sc_port_base &port, const char *if_typename) {}

    transfer_status read(uint64_t address, T &data, bool shadow = false) {
        if (address < start_address_ || address > end_address_) {
            data = 0;
            return TRANSFER_ERROR;
        }
        data = mem[address - start_address_];
        Trace_event_util _util;
        _util.m_bar_name = __func__;
        e_engine->add_event(this->name(), __func__, "B", _util);
        wait(RAM_READ_LATENCY, SC_NS);
        e_engine->add_event(this->name(), __func__, "E", _util);

        return TRANSFER_OK;
    }

    bool reset() {
        for (uint64_t i = 0; i < (end_address_ - start_address_); i++)
            mem[i] = (T)0;
        return true;
    }

    inline uint64_t start_address() const { return start_address_; }

    inline uint64_t end_address() const { return end_address_; }
};