#include "systemc.h"
#include <tlm>

#include "link/instr/recv_global_mem.h"
#include "memory/dram/Dcachecore.h"
#include "prims/base.h"
#include "prims/comp_prims.h"
#include "utils/memory_utils.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Send_global_memory);

void Send_global_memory::initialize() {
    LOG_ERROR(PRIM) << "Send_global_memory::initialize() not implemented";
}

void Send_global_memory::taskCore(TaskCoreContext &context, string prim_name,
                                 u_int64_t &dram_time, u_int64_t &exu_ops,
                                 u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // std::cout << "[Global Mem]: Send_global_memory::taskCoreDefault"
    //           << std::endl;
    // std::cout << "[Global Mem]: Send_global_memory::tag_id: " << tag_id
    //           << std::endl;
    // // Determine element size (INT8=1 byte, FP16=2 bytes)
    // int elem_bytes = (datatype == DATATYPE::FP16 ? 2 : 1);
    // // Total bytes to send (end_length elements)
    // int byte_count = end_length * elem_bytes;
    // // Remote DRAM byte address = des_offset * element size
    // uint64_t address = static_cast<uint64_t>(des_offset) * elem_bytes;

    // // Create a write transaction (simulate latency only, no data pointer)
    // tlm::tlm_generic_payload trans;
    // trans.set_command(tlm::TLM_WRITE_COMMAND);
    // trans.set_address(address);
    // trans.set_data_ptr(nullptr);
    // trans.set_data_length(byte_count);
    // trans.set_streaming_width(byte_count);
    // trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

    // // Dispatch the transaction via the DRAM interface
    // sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    // context.nb_global_memif->socket->b_transport(trans, delay);


    // std::cout << "[Global Mem]: Send_global_memory::end " << std::endl;
    // // Return the simulated write latency in nanoseconds
    // return static_cast<int>(delay.to_seconds() * 1e9);

    LOG_ERROR(PRIM) << "Send_global_memory not implemented";
}