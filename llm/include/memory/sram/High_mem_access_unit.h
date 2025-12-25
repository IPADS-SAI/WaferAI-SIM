#pragma once
#include "systemc.h"

#include "defs/const.h"
#include "memory/mem_if.h"
#include "trace/Event_engine.h"


class high_bw_mem_access_unit : public sc_module {
public:
    SC_HAS_PROCESS(high_bw_mem_access_unit);
    high_bw_mem_access_unit(const sc_module_name nm, Event_engine *e_engine)
        : sc_module(nm), e_engine_(e_engine) {
        // SC_THREAD(memory_read);
    }

    // void memory_read();

public:
    sc_port<multiport_read_if<sc_bv<SRAM_BITWIDTH>, SRAM_BANKS>> mem_read_port;
    Event_engine *e_engine_;
};