#pragma once
#include "systemc.h"
#include <vector>
#include <functional>
#include "memory/sram/High_mem_access_unit.h"
#include "unit_module/sram_manager/sram_manager.h"

using namespace sc_core;
using namespace sc_dt;

using AllocationID = float;


class SRAMWriteModule : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(SRAMWriteModule);


    // 完成通知事件（供外部使用）
    sc_event* e_done;

    // 构造函数
    SRAMWriteModule(sc_core::sc_module_name name,sc_event* e_done):
    sc_module(name)
    {
        SC_THREAD(write_thread);
        this->e_done = e_done;
        // 不敏感于任何信号，由内部事件驱动
    }

    // 触发写操作的接口函数（可被其他模块调用）
    void trigger_write(high_bw_mem_access_unit* hmau,  SramManager *sram_manager,int dma_read_count, int sram_addr_temp, AllocationID alloc_id, int SRAM_BITW, bool use_manager) {
        this->dma_read_count = dma_read_count;
        this->sram_addr_temp = sram_addr_temp;
        this->sram_manager_ = sram_manager;
        this->alloc_id = alloc_id;
        this->use_manager = use_manager;
        this->hmau = hmau;
        this->SRAM_BITW = SRAM_BITW;


        // 启动线程（通过事件）
        start_event.notify(SC_ZERO_TIME);
    }

    // 获取完成事件的引用，用于等待
    // sc_event& get_done_event() {
    //     return e_done;
    // }

private:
    // 线程主函数
    void write_thread() {
        while (true) {
            // 等待被触发
            wait(start_event);


            std::vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
            for (int i = 0; i < SRAM_BANKS; ++i) {
                data_tmp[i] = 0;
            }

            int addr = sram_addr_temp;
            for (int i = 0; i < dma_read_count; ++i) {
                if (i != 0) {
#if USE_SRAM_MANAGER == 1
                    if (use_manager && sram_manager_) {
                        addr = sram_manager_->get_address_with_offset(
                                   alloc_id, addr * SRAM_BITW / 8,
                                   SRAM_BANKS * SRAM_BITW / 8) / (SRAM_BITW / 8);
                    } else {
                        addr += SRAM_BANKS;
                    }
#else
                    addr += SRAM_BANKS;
#endif
                } else {
                    addr = sram_addr_temp;  // 第一次用原始地址
                }

                sc_time elapsed_time;
                // std::cout << "Address of hmau: " << static_cast<void*>(hmau) << std::endl;

                hmau->mem_read_port->multiport_write(addr, data_tmp, elapsed_time);
            }

            // 通知写完成（模拟原 (*context.e_sram).notify()）
            e_done->notify(SC_ZERO_TIME);
        }
    }

    // 内部存储的参数
    SramManager *sram_manager_;
    AllocationID alloc_id;
    int dma_read_count;
    int sram_addr_temp;
    bool use_manager;
    high_bw_mem_access_unit* hmau;
    int SRAM_BITW;
    
    

    // 启动事件
    sc_event start_event;
};