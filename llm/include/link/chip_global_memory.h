#pragma once
#include <string_view>
#include <systemc>
#include <tlm>
#include <vector>

#include <tlm.h>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include "memory/dramsys_wrapper.h"
#include "nlohmann/json.hpp"
#include "trace/Event_engine.h"

#include "defs/const.h"
#include "defs/global.h"
#include "macros/macros.h"
#include "utils/system_utils.h"

using namespace sc_core;
using namespace tlm;

class ChipGlobalMemory : public sc_module {
public:
    int cid;

    // Event signaled on receiving a write transaction
    sc_core::sc_event write_received_event;

    gem5::memory::DRAMSysWrapper *dramSysWrapper;

    tlm_utils::simple_target_socket<ChipGlobalMemory> socket; // Receive
    tlm_utils::simple_initiator_socket<ChipGlobalMemory>
        initiatorSocket; // Send

    ::DRAMSys::Config::Configuration globalMemoryConfig;

    SC_HAS_PROCESS(ChipGlobalMemory);

    ChipGlobalMemory(const sc_module_name &n, std::string_view configuration,
                     std::string_view resource_directory)
        : sc_module(n),
          socket("chip_global_target_socket"),
          initiatorSocket("chip_global_initiator_socket"),
          globalMemoryConfig(
              ::DRAMSys::Config::from_path(configuration, resource_directory)) {
        dramSysWrapper = new gem5::memory::DRAMSysWrapper(
            "GlobalDRAMSysWrapper", globalMemoryConfig, false);
        initiatorSocket.bind(dramSysWrapper->tSocket);

        socket.register_b_transport(this, &ChipGlobalMemory::b_transport);
        socket.register_nb_transport_fw(this,
                                        &ChipGlobalMemory::nb_transport_fw);
        initiatorSocket.register_nb_transport_bw(
            this, &ChipGlobalMemory::nb_transport_bw);
    }

    ~ChipGlobalMemory() { delete dramSysWrapper; }

    void b_transport(tlm_generic_payload &trans, sc_core::sc_time &delay) {
        // Notify that a write transaction has been received
        write_received_event.notify(sc_core::SC_ZERO_TIME);
        initiatorSocket->b_transport(trans, delay);
    }

    tlm::tlm_sync_enum nb_transport_fw(tlm::tlm_generic_payload &trans,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay) {
        return initiatorSocket->nb_transport_fw(trans, phase, delay);
    }

    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload &payload,
                                       tlm::tlm_phase &phase,
                                       sc_time &bwDelay) {
        socket->nb_transport_bw(payload, phase, bwDelay);
        return tlm::TLM_ACCEPTED;
    }
};