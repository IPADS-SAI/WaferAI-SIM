#pragma once
#include "defs/const.h"
#include "memory/mem_if.h"
#include "systemc.h"
#include "trace/Event_engine.h"


class mem_access_unit : public sc_module {
public:
    SC_HAS_PROCESS(mem_access_unit);
    mem_access_unit(const sc_module_name nm, Event_engine *e_engine)
        : sc_module(nm), e_engine_(e_engine) {}

public:
    sc_port<mem_read_time_if<sc_bv<SRAM_BITWIDTH>>> mem_read_port;
    sc_port<mem_write_time_if<sc_bv<SRAM_BITWIDTH>>> mem_write_port;
    Event_engine *e_engine_;
};
