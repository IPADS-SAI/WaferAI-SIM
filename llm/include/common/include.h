#pragma once
#include <string>
#include "systemc.h"

#include "macros/macros.h"
#include "defs/enums.h"
#include "memory/dram/Dcachecore.h"
#include "memory/dram/GPUNB_DcacheIF.h"
#include "memory/dram/NB_DcacheIF.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "memory/sram/High_mem_access_unit.h"
#include "memory/sram/Mem_access_unit.h"
#include "memory/sram_writer.h"
#include "link/nb_global_memif_v2.h"

using namespace std;

class AddrDatapassLabel {
public:
    string indata[MAX_SPLIT_NUM];
    string outdata;
    AddrDatapassLabel() {
        for (int i = 0; i < MAX_SPLIT_NUM; i++) {
            indata[i] =
                UNSET_LABEL; // 读入sram的输入数据标签，绝大多数情况下只使用第一个元素
        }

        outdata = UNSET_LABEL; // 写回sram的输出数据标签
    }

    AddrDatapassLabel(string indata_v, string outdata_v) : outdata(outdata_v) {
        indata[0] = indata_v;
    }
};


class Stage {
public:
    int req_id;
    PD_PHASE type;
    int token_num;
    int total_iter; // 用于prefill

    Stage() {}
    Stage(int id, PD_PHASE type, int token) : req_id(id), type(type), token_num(token) {}
};

// 用于传入taskCore
class TaskCoreContext {
public:
    int cid;

    mem_access_unit *mau;
    high_bw_mem_access_unit *hmau;
    mem_access_unit *temp_mau;
    high_bw_mem_access_unit *temp_hmau;
    int *sram_addr;
    SramManager *sram_manager_;
    AllocationID alloc_id_;
    sc_event *s_nbdram;
    sc_event *e_nbdram;
    sc_event *s_sram;
    sc_event *e_sram;
    SRAMWriteModule *sram_writer;
    int loop_cnt;
    uint64_t MaxDramAddr; // 当前核最大的 dram 地址
    unsigned int defaultDataLength;

    NB_GlobalMemIF *nb_global_memif;
    sc_event *start_global_event;
    sc_event *end_global_event;

    void SetGlobalMemIF(NB_GlobalMemIF *nb_global_memif,
                        sc_event *start_global_event,
                        sc_event *end_global_event) {
        this->nb_global_memif = nb_global_memif;
        this->start_global_event = start_global_event;
        this->end_global_event = end_global_event;
    }

#if USE_NB_DRAMSYS == 1

    NB_DcacheIF *nb_dcache;


#else
    DcacheCore *wc;
#endif
#if USE_L1L2_CACHE == 1
    // tlm_utils::simple_initiator_socket<Processor> *cache_socket;
    GPUNB_dcacheIF *gpunb_dcache_if;
    Event_engine *event_engine;
    sc_event *start_nb_gpu_dram_event;
    sc_event *end_nb_gpu_dram_event;
#endif
#if USE_NB_DRAMSYS == 1
    // 构造函数
    TaskCoreContext(mem_access_unit *mau, high_bw_mem_access_unit *hmau,
                    mem_access_unit *temp_mau,
                    high_bw_mem_access_unit *temp_hmau, int *sram_addr,
                    sc_event *s_nbdram, sc_event *e_nbdram,
                    NB_DcacheIF *nb_dcache, SramManager *sram_manager,
                    uint64_t MaxDramAddr, unsigned int defaultDataLength)
        : mau(mau),
          hmau(hmau),
          temp_mau(temp_mau),
          temp_hmau(temp_hmau),
          sram_addr(sram_addr),
          s_nbdram(s_nbdram),
          e_nbdram(e_nbdram),
          nb_dcache(nb_dcache),
          sram_manager_(sram_manager),
          MaxDramAddr(MaxDramAddr),
          defaultDataLength(defaultDataLength) {}
    TaskCoreContext(mem_access_unit *mau, high_bw_mem_access_unit *hmau,
                    mem_access_unit *temp_mau,
                    high_bw_mem_access_unit *temp_hmau, int *sram_addr,
                    sc_event *s_nbdram, sc_event *e_nbdram,
                    NB_DcacheIF *nb_dcache, SramManager *sram_manager,
                    int loop_cnt, uint64_t MaxDramAddr,
                    unsigned int defaultDataLength)
        : mau(mau),
          hmau(hmau),
          temp_mau(temp_mau),
          temp_hmau(temp_hmau),
          sram_addr(sram_addr),
          s_nbdram(s_nbdram),
          e_nbdram(e_nbdram),
          nb_dcache(nb_dcache),
          loop_cnt(loop_cnt),
          sram_manager_(sram_manager),
          MaxDramAddr(MaxDramAddr),
          defaultDataLength(defaultDataLength) {}
#else
    // 构造函数
    TaskCoreContext(DcacheCore *wc, mem_access_unit *mau,
                    high_bw_mem_access_unit *hmau, mem_access_unit *temp_mau,
                    high_bw_mem_access_unit *temp_hmau, int *sram_addr,
                    sc_event *s_nbdram, sc_event *e_nbdram,
                    SramManager *sram_manager, int loop_cnt,
                    uint64_t MaxDramAddr, unsigned int defaultDataLength)
        : wc(wc),
          mau(mau),
          hmau(hmau),
          temp_mau(temp_mau),
          temp_hmau(temp_hmau),
          sram_addr(sram_addr),
          s_nbdram(s_nbdram),
          e_nbdram(e_nbdram),
          sram_manager_(sram_manager),
          loop_cnt(loop_cnt),
          MaxDramAddr(MaxDramAddr),
          defaultDataLength(defaultDataLength) {}
#endif

#if USE_L1L2_CACHE == 1
    TaskCoreContext(mem_access_unit *mau, high_bw_mem_access_unit *hmau,
                    mem_access_unit *temp_mau,
                    high_bw_mem_access_unit *temp_hmau, int *sram_addr,
                    sc_event *s_nbdram, sc_event *e_nbdram,
                    NB_DcacheIF *nb_dcache, int loop_cnt,
                    SramManager *sram_manager,
                    sc_event *start_nb_gpu_dram_event,
                    sc_event *end_nb_gpu_dram_event, uint64_t MaxDramAddr,
                    unsigned int defaultDataLength)
        : mau(mau),
          hmau(hmau),
          temp_mau(temp_mau),
          temp_hmau(temp_hmau),
          sram_addr(sram_addr),
          s_nbdram(s_nbdram),
          e_nbdram(e_nbdram),
          nb_dcache(nb_dcache),
          loop_cnt(loop_cnt),
          sram_manager_(sram_manager),
          start_nb_gpu_dram_event(start_nb_gpu_dram_event),
          end_nb_gpu_dram_event(end_nb_gpu_dram_event),
          MaxDramAddr(MaxDramAddr),
          defaultDataLength(defaultDataLength) {}
#endif
};