#pragma once
#include "systemc.h"
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include "trace/Event_engine.h"

class DcacheCore : public sc_module {
public:
    int cid;
    Event_engine *event_engine;

    tlm_utils::simple_initiator_socket<DcacheCore> isocket;

    SC_HAS_PROCESS(DcacheCore);
    DcacheCore(const sc_module_name &n, Event_engine *event_engine)
        : isocket("Dcache_socket") {};
    ~DcacheCore() {}
};