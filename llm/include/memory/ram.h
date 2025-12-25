#pragma once
#ifndef _RAM_H
#define _RAM_H

#include "systemc.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "defs/const.h"
#include "macros/macros.h"
#include "mem_if.h"
#include "trace/Event_engine.h"


template <class T> class Ram : public sc_channel, public ram_if<T> {
private:
    T *mem;
    uint64_t start_address_;
    uint64_t end_address_;
    bool *valid;

public:
    static int random_access_times;
    static float energy_consumption;
    static int total_access_latency;
    static float area;
    Event_engine *e_engine;
    bool ram_write_busy;
    bool ram_read_busy;
    string name;

public:
    Ram(sc_module_name name_, uint64_t start_address, uint64_t end_address,
        Event_engine *_e_engine)
        : name(name_),
          start_address_(start_address),
          end_address_(end_address) {
        // sc_assert(end_address_ >= m_start_addresss);
#if DUMMY_SRAM == 1


#else
        mem = new T[end_address_ - start_address_];
#endif
#if DUMMY_SRAMV == 0
        valid = new bool[end_address_ - start_address_];
        for (auto i = start_address_; i < end_address_; i++) {
            valid[i - start_address_] = 0;
        }
#endif
        e_engine = _e_engine;

        auto bits = (end_address_ - start_address_) * sizeof(T) * 8;
        area = 0.001 * std::pow(0.028, 2.07) * std::pow(bits, 0.9) *
                   std::pow(2, 0.7) +
               0.0048;
    }
    ~Ram() {
        if (mem) {
            delete[] mem;
        }
#if DUMMY_SRAMV == 0
        if (valid) {
            delete[] valid;
        }
#endif
    }

    virtual void register_port(sc_port_base &port, const char *if_typename) {}

    transfer_status read(uint64_t address, T &data, bool shadow = false) {
        if (address < start_address_ || address > end_address_) {
            data = 0;
            return TRANSFER_ERROR;
        }
        ram_read_busy = 1;
#if DUMMY_SRAM == 1
        data = 0;
#else
        data = mem[address - start_address_];
#endif
#if VERBOSE_TRACE == 1
        e_engine->add_event(this->name, __func__, "B",
                            Trace_event_util("read"));
#endif
        if (!shadow) {
            wait(RAM_READ_LATENCY, SC_NS);
#if VERBOSE_TRACE == 1
            e_engine->add_event(this->name, __func__, "E",
                                Trace_event_util("read"));
#endif
        } else {
#if VERBOSE_TRACE == 1
            e_engine->add_event(this->name, __func__, "E",
                                Trace_event_util("read"),
                                sc_time(RAM_READ_LATENCY, SC_NS));
#endif
        }

        random_access_times++;
        total_access_latency += RAM_READ_LATENCY;
        energy_consumption += RAM_READ_ENERGY;
        ram_read_busy = 0;
        return TRANSFER_OK;
    }

    transfer_status write(uint64_t address, T &data, bool force_write = 1,
                          bool shadow = true) {
        if (address < start_address_ || address > end_address_) {
            return TRANSFER_ERROR;
        }
        ram_write_busy = 1;
#if DUMMY_SRAM == 1

#else
        mem[address - start_address_] = data;
#endif
#if DUMMY_SRAMV == 0
        if (force_write == 1) {
            valid[address - start_address_] = 1;
        } else {
            if (valid[address - start_address_] == 1) {
                assert(false && "Ram valid error.");
                cout << "Ram valid error" << endl;
            } else {
                valid[address - start_address_] = 1;
            }
        }
#endif
        // std::cout << this->name << __func__;
#if VERBOSE_TRACE == 1
        e_engine->add_event(this->name, __func__, "B",
                            Trace_event_util("write"));
#endif
        if (!shadow) {
            wait(RAM_WRITE_LATENCY, SC_NS);
#if VERBOSE_TRACE == 1
            e_engine->add_event(this->name, __func__, "E",
                                Trace_event_util("write"));
#endif
        } else {
#if VERBOSE_TRACE == 1
            e_engine->add_event(this->name, __func__, "E",
                                Trace_event_util("write"),
                                sc_time(RAM_WRITE_LATENCY, SC_NS));
#endif
        }


        random_access_times++;
        total_access_latency += RAM_WRITE_LATENCY;
        energy_consumption += RAM_WRITE_ENERGY;
        int valid_tmp = 0;
        float valid_perc = 0;
#if DUMMY_SRAMV == 0
        for (auto i = start_address_; i < end_address_; i++) {
            if (valid[i - start_address_] == 1) {
                valid_tmp++;
            };
        }
#endif
        valid_perc = (float)valid_tmp / (end_address_ - start_address_);
        string color;
        if (valid_perc > Men_usage_thre) {
            color = "cq_build_attempt_failed";
        } else {
            color = "cq_build_attempt_runnig";
        }
        // std::cout << sc_get_current_process_b()->name() << " " <<
        // sc_time_stamp() << " aa  " << valid_tmp << std::endl;
#if VERBOSE_TRACE == 1
        e_engine->add_event(this->name, "Mem Usage", "C",
                            Trace_event_util(valid_perc, color),
                            sc_time(shadow ? RAM_WRITE_LATENCY : 0, SC_NS));
#endif
        ram_write_busy = 0;
        return TRANSFER_OK;
    }


    transfer_status clear(uint64_t address, T &data) {
        if (address < start_address_ || address > end_address_) {
            return TRANSFER_ERROR;
        }
#if DUMMY_SRAMV == 0
        valid[address - start_address_] = 0;
#endif
        return TRANSFER_OK;
    }
    bool reset() {
#if DUMMY_SRAM == 1

#else
        for (uint64_t i = 0; i < (end_address_ - start_address_); i++)
            mem[i] = (T)0;
#endif
        return true;
    }

    inline uint64_t start_address() const { return start_address_; }

    inline uint64_t end_address() const { return end_address_; }
};

template <class T> int Ram<T>::random_access_times = 0;

template <class T> float Ram<T>::area = 0;

template <class T> float Ram<T>::energy_consumption = 0;

template <class T> int Ram<T>::total_access_latency = 0;


#endif
