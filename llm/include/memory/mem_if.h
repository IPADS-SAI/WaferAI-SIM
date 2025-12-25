#ifndef _MEM_IF_H
#define _MEM_IF_H
#include "systemc.h"
#include <vector>

enum transfer_status { TRANSFER_OK = 0, TRANSFER_ERROR };

using namespace std;

template <class T> class mem_clear_if : virtual public sc_interface {
public:
    virtual transfer_status clear(uint64_t address, T &data) = 0;
};

template <class T> class mem_read_if : virtual public sc_interface {
public:
    virtual transfer_status read(uint64_t address, T &data,
                                 bool shadow = false) = 0;
};

template <class T> class mem_write_if : virtual public sc_interface {
public:
    virtual transfer_status write(uint64_t address, T &data,
                                  bool force_write = 1, bool shadow = true) = 0;
};

template <class T> class mem_read_time_if : virtual public sc_interface {
public:
    virtual transfer_status read(uint64_t address, T &data,
                                 sc_time &elapsed_time,
                                 bool shadow = false) = 0;
};

template <class T> class mem_write_time_if : virtual public sc_interface {
public:
    virtual transfer_status write(uint64_t address, T &data,
                                  sc_time &elapsed_time, bool force_write = 1,
                                  bool shadow = true) = 0;
};

template <class T, unsigned M>
class multiport_read_if : virtual public sc_interface {
public:
    virtual transfer_status multiport_read(uint64_t address_in_high_bitwidth,
                                           vector<T> &data,
                                           sc_time &elapsed_time) = 0;
    virtual transfer_status multiport_write(uint64_t address_in_high_bitwidth,
                                            vector<T> &data,
                                            sc_time &elapsed_time) = 0;
};

class reset_if : virtual public sc_interface {
public:
    virtual bool reset() = 0;
};

template <class T>
class ram_if : virtual public mem_write_if<T>,
               virtual public mem_read_if<T>,
               virtual public reset_if,
               virtual public mem_clear_if<T> {
public:
    virtual uint64_t start_address() const = 0;
    virtual uint64_t end_address() const = 0;
};


template <class T>
class ram_time_if : virtual public mem_write_time_if<T>,
                    virtual public mem_read_time_if<T>,
                    virtual public reset_if,
                    virtual public mem_clear_if<T> {
public:
    virtual uint64_t start_address() const = 0;
    virtual uint64_t end_address() const = 0;
};


template <class T>
class rom_if : virtual public mem_read_if<T>, virtual public reset_if {
public:
    virtual uint64_t start_address() const = 0;
    virtual uint64_t end_address() const = 0;
};

template <class T> class burst_ram_if : virtual public ram_if<T> {
public:
    virtual transfer_status burst_read(uint64_t address, uint64_t length,
                                       vector<T> &data) = 0;
    virtual transfer_status burst_write(uint64_t address, uint64_t length,
                                        vector<T> &data,
                                        uint64_t force_write = 1) = 0;
};

#endif
