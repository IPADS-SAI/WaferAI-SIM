// #pragma once

// #include <systemc>
// #include <tlm>
// #include <tlm_utils/peq_with_cb_and_phase.h>
// #include <tlm_utils/simple_initiator_socket.h>
// #include <tlm_utils/simple_target_socket.h>
// #include <queue>

// #include "memory/dramsys_wrapper.h"
// #include "trace/Event_engine.h"
// #include "nlohmann/json.hpp"
// #include "defs/const.h"
// #include "defs/global.h"
// #include "macros/macros.h"
// #include "utils/system_utils.h"

// using namespace sc_core;
// using namespace tlm;

// class ChipGlobalMemIF_NB : public sc_module {
//     SC_HAS_PROCESS(ChipGlobalMemIF_NB);
    
//     gem5::memory::DRAMSysWrapper* dramSysWrapper;
//     tlm_utils::simple_target_socket<ChipGlobalMemIF_NB> socket;
//     tlm_utils::simple_initiator_socket<ChipGlobalMemIF_NB> initiatorSocket;

    

// };

#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include "memory/MemoryManager_v2.h"

using namespace sc_core;
using namespace tlm;

class NB_ChipGlobalMemoryIF : public sc_module {
public:
    // TLM接口
    tlm_utils::simple_initiator_socket<NB_ChipGlobalMemoryIF> socket;
    
    // 事件系统
    sc_event* start_dram_access;   // 全局内存访问启动事件
    sc_event* end_dram_access;     // 全局内存访问完成事件
    sc_event  next_transaction_ev; // 下一事务触发事件
    
    // 事务管理
    tlm_utils::peq_with_cb_and_phase<NB_ChipGlobalMemoryIF> peq;
    MemoryManager_v2 mm;
    
    // 流控参数
    const unsigned int MAX_PENDING = 16;          // 最大挂起事务数
    unsigned int pending_reads = 0;               // 当前读事务数
    unsigned int pending_writes = 0;              // 当前写事务数
    bool flow_control_blocked = false;            // 流控阻塞标志
    
    // 统计信息
    uint64_t completed_transactions = 0;          // 已完成事务计数
    uint64_t total_transactions = 0;              // 总事务数
    
    SC_HAS_PROCESS(NB_ChipGlobalMemoryIF);

    NB_ChipGlobalMemoryIF(sc_module_name name, 
                        sc_event* start_ev, 
                        sc_event* end_ev)
        : sc_module(name),
        start_dram_access(start_ev),
        end_dram_access(end_ev),
        peq(this, &NB_ChipGlobalMemoryIF::peq_callback),
        socket("global_mem_socket"),
        mm(false)
    {
        SC_THREAD(transaction_engine);
        socket.register_nb_transport_bw(this, &NB_ChipGlobalMemoryIF::nb_transport_bw);
    }

    // 非阻塞反向路径
    tlm_sync_enum nb_transport_bw(tlm_generic_payload& trans, 
                                tlm_phase& phase, 
                                sc_time& delay) 
    {
        peq.notify(trans, phase, delay);
        return TLM_ACCEPTED;
    }

    // 配置传输参数
    void configure(uint64_t base_addr, 
                 uint32_t total_trans, 
                 uint32_t burst_length) 
    {
        base_address = base_addr;
        total_transactions = total_trans;
        burst_size = burst_length;
        current_transaction = 0;
    }

private:
    // 事务状态
    uint64_t base_address = 0;
    uint32_t burst_size = 64;      // 默认突发长度64字节
    uint32_t current_transaction = 0;
    
    // PEQ回调处理
    void peq_callback(tlm_generic_payload& trans, const tlm_phase& phase) 
    {
        sc_time delay = SC_ZERO_TIME;
        
        switch(phase) {
        case BEGIN_RESP: {
            // 更新统计信息
            completed_transactions++;
            
            // 释放事务资源
            trans.release();
            
            // 流控更新
            if(trans.get_command() == TLM_READ_COMMAND) {
                pending_reads--;
            } else {
                pending_writes--;
            }
            
            // 解除流控阻塞
            if(flow_control_blocked && can_send()) {
                flow_control_blocked = false;
                next_transaction_ev.notify(delay);
            }
            
            // 完成检测
            if(completed_transactions >= total_transactions) {
                end_dram_access->notify(delay);
            }
            break;
        }
        default:
            SC_REPORT_FATAL(name(), "Invalid phase received");
        }
    }

    // 事务生成引擎
    void transaction_engine() 
    {
        while(true) {
            wait(*start_dram_access);  // 等待访问触发
            
            while(current_transaction < total_transactions) {
                // 流控检查
                if(!can_send()) {
                    flow_control_blocked = true;
                    wait(next_transaction_ev);
                    continue;
                }
                
                // 生成事务
                tlm_generic_payload& trans = mm.allocate(burst_size);
                trans.acquire();
                
                setup_transaction(trans);
                
                // 发送事务
                sc_time delay = SC_ZERO_TIME;
                tlm_phase phase = BEGIN_REQ;
                tlm_sync_enum status;
                
                status = socket->nb_transport_fw(trans, phase, delay);
                
                // 处理返回值
                if(status == TLM_UPDATED) {
                    peq.notify(trans, phase, delay);
                } else if(status == TLM_COMPLETED) {
                    mm.free(&trans);
                    completed_transactions++;
                }
                
                // 更新状态
                if(trans.get_command() == TLM_READ_COMMAND) {
                    pending_reads++;
                } else {
                    pending_writes++;
                }
                
                current_transaction++;
                
                // 添加请求间隔
                wait(delay);
            }
            
            // 等待最后一笔事务完成
            while(completed_transactions < total_transactions) {
                wait(next_transaction_ev);
            }
        }
    }

    // 事务参数设置
    void setup_transaction(tlm_generic_payload& trans) 
    {
        uint64_t address = base_address + current_transaction * burst_size;
        
        trans.set_address(address);
        trans.set_data_length(burst_size);
        trans.set_streaming_width(burst_size);
        trans.set_command(current_transaction % 2 ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
        trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
    }

    // 流控判断
    bool can_send() const 
    {
        return (pending_reads + pending_writes) < MAX_PENDING;
    }
};