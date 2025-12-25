#include "utils/memory_utils.h"
#include "common/system.h"
#include "defs/const.h"
#include "defs/global.h"
#include "macros/macros.h"
#include "memory/gpu/GPU_L1L2_Cache.h"
#include "utils/print_utils.h"
#include "utils/system_utils.h"


#include "systemc.h"
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>


void sram_write(TaskCoreContext &context, int dma_read_count,
                int sram_addr_temp, AllocationID alloc_id, bool use_manager) {
    //     auto hmau = context.hmau;
    //     auto sram_manager_ = context.sram_manager_;
    //     vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    //     for (int i = 0; i < SRAM_BANKS; i++) {
    //         data_tmp[i] = 0;
    //     }
    //     for (int i = 0; i < dma_read_count; i++) {
    //             if (i != 0) {
    // #if USE_SRAM_MANAGER == 1
    //                 if (use_manager == true) {
    //                     sram_addr_temp =
    //                         sram_manager_->get_address_with_offset(
    //                             alloc_id, sram_addr_temp * sram_bitw / 8,
    //                             SRAM_BANKS * sram_bitw / 8) /
    //                         (sram_bitw / 8);
    //                 } else {
    //                     sram_addr_temp = sram_addr_temp + SRAM_BANKS;
    //                 }
    // #else
    //                 sram_addr_temp = sram_addr_temp + SRAM_BANKS;
    // #endif
    //             }
    //             sc_time elapsed_time;
    //             hmau->mem_read_port->multiport_write(sram_addr_temp,
    //             data_tmp,
    //                                                  elapsed_time);
    //         }
    //     (*context.e_sram).notify(SC_ZERO_TIME);
}
// revise context.sram_addr value
void sram_first_write_generic(TaskCoreContext &context, int data_size_in_byte,
                              u_int64_t global_addr, u_int64_t &dram_time,
                              float *dram_start, std::string label_name,
                              bool use_manager,
                              SramPosLocator *sram_pos_locator,
                              bool dummy_alloc, bool add_dram_addr) {

    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;

    int dma_read_count =
        max(1, data_size_in_byte * 8 / (sram_bitw * SRAM_BANKS));
    int byte_residue = max(0, data_size_in_byte * 8 -
                                  dma_read_count * (sram_bitw * SRAM_BANKS));
    int single_read_count = CeilingDivision(byte_residue, sram_bitw);

    int data_bits = data_size_in_byte * 8;
    assert((SRAM_BLOCK_SIZE * 8) % sram_bitw == 0 &&
           "SRAM_BLOCK_SIZE * 8 must be a multiple of SRAM_BITWIDTH");
    int alignment = std::max(sram_bitw, SRAM_BLOCK_SIZE * 8);

    int aligned_data_bits = static_cast<int>(std::ceil(
                                static_cast<double>(data_bits) / alignment)) *
                            alignment;
    int aligned_data_byte = aligned_data_bits / 8;

    u_int64_t inp_global_addr = global_addr;
    int left_byte = aligned_data_byte - data_size_in_byte;


#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    auto sram_addr = context.sram_addr;
    auto sram_addr_temp = *sram_addr;
    auto sram_manager_ = context.sram_manager_;
#if USE_NB_DRAMSYS == 1
    auto nb_dcache = context.nb_dcache;
#endif
    auto s_nbdram = context.s_nbdram;
    auto e_nbdram = context.e_nbdram;

    sc_event_and_list ram_e;
    if (!SPEC_USE_BEHA_SRAM) {
        ram_e &= *context.e_nbdram;
        ram_e &= *context.e_sram;
    }

#if USE_GLOBAL_DRAM == 1

    inp_global_addr =
        (global_addr / 64) * 64; // 向下取整到dram 取址的整数倍，这里是32
    int end_addr = global_addr + data_size_in_byte;
    int end_global_addr = ((end_addr + 63) / 64) * 64; // 尾地址向上取整

    int aligned_data_size_in_byte = end_global_addr - inp_global_addr;


    auto gpunb_dcache_if = context.gpunb_dcache_if;

    s_nbdram = context.start_nb_gpu_dram_event;
    e_nbdram = context.end_nb_gpu_dram_event;

    u_int64_t in_dcacheline_g = 0;
    int cache_lines_g = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count_g =
        CeilingDivision(aligned_data_size_in_byte * 8, cache_lines_g);
#endif


    // std::vector<std::pair<u_int64_t, sc_time>> addr_time_pairs;
    // // 获取当前线程的名字
    // std::string thread_name = sc_core::sc_get_current_process_b()->name();
    // std::string filename = "addr_time_data_" + thread_name + ".txt";
    u_int64_t in_dcacheline = 0;
    int cache_lines = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count = SRAM_BANKS * sram_bitw / cache_lines;
    assert(cache_count > 0 &&
           "cache_count must be larger than 0, check SRAM_BANKS and sram_bitw");
    int sram_time = 0;


    sc_time start_first_write_time = sc_time_stamp();

    AllocationID alloc_id = 0;
#if USE_SRAM_MANAGER == 1

    if (use_manager == true) {
        SizeWAddr swa(aligned_data_byte, inp_global_addr);
#if ASSERT == 1
        assert(sram_pos_locator->validateTotalSize() &&
               "sram_pos_locator is not equal sram_manager");
#endif
        AddrPosKey inp_key = AddrPosKey(swa);
        u_int64_t dram_time_tmp = 0;
        sram_pos_locator->addPair(label_name, inp_key, context, dram_time_tmp,
                                  add_dram_addr);
        // 使用 SramManager 分配内存
        // sram_manager_->display_status();
        alloc_id = sram_manager_->allocate(aligned_data_byte);
        assert(alloc_id > 0 && "alloc_id must larger than 0 ");
        context.alloc_id_ = alloc_id;
        inp_key = AddrPosKey(context.alloc_id_, aligned_data_byte);
        inp_key.left_byte = left_byte;
        sram_pos_locator->addPair(label_name, inp_key, false);

        LOG_DEBUG(MEMORY) << "SRAM Manager Allocation - Core " << context.cid
                          << ", Label: " << label_name
                          << ", AllocationID: " << alloc_id
                          << ", Size: " << aligned_data_byte << " bytes";

        if (alloc_id == 0) {
            LOG_ERROR(MEMORY) << "Failed to allocate " << aligned_data_byte
                              << " bytes from SRAM.";
        } else {
            LOG_DEBUG(MEMORY_DEBUG)
                << "Successfully allocated " << aligned_data_byte
                << " bytes with AllocationID: " << alloc_id;

            // 获取实际地址用于后续操作
            sram_addr_temp = sram_manager_->get_address_index(alloc_id);
            LOG_DEBUG(MEMORY_DEBUG) << "SRAM Address Index: " << sram_addr_temp;
        }
    }
    assert(sram_pos_locator->validateTotalSize() &&
           "sram_pos_locator is not equal sram_manager");
#endif
    if (dummy_alloc == false) {
#if USE_NB_DRAMSYS == 1
#if DRAM_BURST_BYTE > 0
        assert(DRAM_BURST_BYTE > context.defaultDataLength);
#endif

#if USE_GLOBAL_DRAM == 1
        // assert(inp_global_addr!=0);

        gpunb_dcache_if->reconfigure(inp_global_addr, cache_count_g,
                                     cache_lines_g, 0);
        sc_time start_nbdram = sc_time_stamp();
        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "read_gpu", "B",
                                        Trace_event_util("read_gpu"));

        wait(*e_nbdram);
        sc_time end_nbdram = sc_time_stamp();
        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "read_gpu", "E",
                                        Trace_event_util("read_gpu"));

#else
        if (!SPEC_USE_BEHA_SRAM) {
            context.sram_writer->trigger_write(
                hmau, sram_manager_, dma_read_count, sram_addr_temp, alloc_id,
                sram_bitw, use_manager);
        }

        if (!SPEC_USE_BEHA_DRAM) {
            nb_dcache->reconfigure(inp_global_addr, dma_read_count, cache_count,
                                   cache_lines, 0);
        }

        sc_time start_nbdram = sc_time_stamp();
        LOG_DEBUG(MEMORY) << "Core " << context.cid
                          << " start writing data from DRAM to SRAM";

        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "R_Dram", "B",
                                        Trace_event_util("R_Dram"));
        if (!SPEC_USE_BEHA_DRAM) {
            if (SPEC_USE_BEHA_SRAM)
                wait(*e_nbdram);
            else
                wait(ram_e);
        } else {
            auto require_byte = dma_read_count * cache_count * cache_lines / 8;
            float need_NS = (float)require_byte / HW_BEHA_DRAM_UTIL /
                            (15.0 * GetCoreHWConfig(context.cid)->dram_bw / 8);
            int need_cycles = need_NS;
            wait(need_cycles, SC_NS);
        }
        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "R_Dram", "E",
                                        Trace_event_util("R_Dram"));
        sc_time end_nbdram = sc_time_stamp();
        LOG_DEBUG(MEMORY) << "Core " << context.cid
                          << " end writing data from DRAM to SRAM";
#endif
        u_int64_t nbdram_time = (end_nbdram - start_nbdram).to_seconds() * 1e9;

        LOG_DEBUG(MEMORY_DEBUG)
            << "Core " << context.cid << " nbdram time: " << nbdram_time
            << " dma_read_count: " << dma_read_count
            << " cache_count: " << cache_count
            << " cache_lines: " << cache_lines;

        if (SPEC_USE_BEHA_SRAM) {
            for (int i = 0; i < dma_read_count; i++) {
                if (i != 0) {
#if USE_SRAM_MANAGER == 1
                    if (use_manager == true) {
                        sram_addr_temp =
                            sram_manager_->get_address_with_offset(
                                alloc_id, sram_addr_temp * sram_bitw / 8,
                                SRAM_BANKS * sram_bitw / 8) /
                            (sram_bitw / 8);
                    } else {
                        sram_addr_temp = sram_addr_temp + SRAM_BANKS;
                    }
#else
                    sram_addr_temp = sram_addr_temp + SRAM_BANKS;
#endif
                }

                sram_time += RAM_WRITE_LATENCY;
            }
            if (nbdram_time < sram_time) {
                wait(sram_time - nbdram_time, SC_NS);
            }
        }
#else
        u_int64_t nbdram_time = 0;
        for (int i = 0; i < dma_read_count; i++) {
            for (int j = 0; j < cache_count; j++) {
                in_dcacheline = inp_global_addr;
                tlm::tlm_generic_payload trans;
                trans.set_address(in_dcacheline);
                trans.set_data_length(cache_lines / 8);
                trans.set_streaming_width(cache_lines / 8);
                trans.set_command(tlm::TLM_READ_COMMAND);
#if DUMMY == 1
                trans.set_data_ptr(
                    reinterpret_cast<unsigned char *>((void *)0));
#else
                trans.set_data_ptr(
                    reinterpret_cast<unsigned char *>(dram_start));
#endif
                trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
                sc_time delay = sc_time(0, SC_NS);
                wc->isocket->b_transport(trans, delay);
                u_int64_t timer = delay.to_seconds() * 1e9;
                nbdram_time += timer;
                // wait(delay);
                // addr_time_pairs.emplace_back(inp_global_addr,
                // sc_time_stamp());
                inp_global_addr += cache_lines;
            }

            sram_time = sram_time += RAM_WRITE_LATENCY;

            sram_addr_temp = sram_addr_temp + SRAM_BANKS;
        }

        wait(nbdram_time, SC_NS);
        if (nbdram_time < sram_time) {
            wait(sram_time - nbdram_time, SC_NS);
        }

#endif

        if (single_read_count > 0) {
#if USE_NB_DRAMSYS == 1
#if USE_GLOBAL_DRAM == 0
            // 配置 DRAM 读取
            if (!SPEC_USE_BEHA_DRAM) {
                nb_dcache->reconfigure(inp_global_addr + cache_lines *
                                                             cache_count *
                                                             dma_read_count,
                                       1, cache_count, cache_lines, 0);
            }

            // 触发 SRAM 写入（如果不使用行为模型）
            if (!SPEC_USE_BEHA_SRAM) {
                context.sram_writer->trigger_write(
                    hmau, sram_manager_, single_read_count, sram_addr_temp,
                    alloc_id, sram_bitw, use_manager);
            }

            start_nbdram = sc_time_stamp();
            context.event_engine->add_event("Core " + ToHexString(context.cid),
                                            "R_Dram", "B",
                                            Trace_event_util("R_Dram"));

            // 等待 DRAM 和 SRAM 完成
            if (!SPEC_USE_BEHA_DRAM) {
                if (SPEC_USE_BEHA_SRAM)
                    wait(*e_nbdram);
                else
                    wait(ram_e);
            } else {
                // 使用行为模型计算 DRAM 时间
                auto require_byte = cache_count * cache_lines / 8;
                float need_NS =
                    (float)require_byte / HW_BEHA_DRAM_UTIL /
                    (15.0 * GetCoreHWConfig(context.cid)->dram_bw / 8);
                int need_cycles = need_NS;
                wait(need_cycles, SC_NS);
            }

            context.event_engine->add_event("Core " + ToHexString(context.cid),
                                            "R_Dram", "E",
                                            Trace_event_util("R_Dram"));
            end_nbdram = sc_time_stamp();

            nbdram_time = (end_nbdram - start_nbdram).to_seconds() * 1e9;

            // 如果使用 SRAM 行为模型，计算 SRAM 写入时间
            if (SPEC_USE_BEHA_SRAM) {
                sram_time = 0;
                for (int i = 0; i < single_read_count; i++) {
                    if (i != 0) {
#if USE_SRAM_MANAGER == 1
                        if (use_manager == true) {
                            sram_addr_temp =
                                sram_manager_->get_address_with_offset(
                                    alloc_id, sram_addr_temp * sram_bitw / 8,
                                    sram_bitw / 8) /
                                (sram_bitw / 8);
                        } else {
                            sram_addr_temp = sram_addr_temp + 1;
                        }
#else
                        sram_addr_temp = sram_addr_temp + 1;
#endif
                    }
                    sram_time += RAM_WRITE_LATENCY;
                }

                // 如果 SRAM 写入时间更长，需要额外等待
                if (nbdram_time < sram_time) {
                    wait(sram_time - nbdram_time, SC_NS);
                }
            }

            LOG_DEBUG(MEMORY_DEBUG)
                << "Core " << context.cid
                << " single_read nbdram time: " << nbdram_time
                << " single_read_count: " << single_read_count
                << " sram_time: " << sram_time;
#endif // USE_GLOBAL_DRAM == 0

#else // USE_NB_DRAMSYS == 0
      // 使用传统的 TLM 传输方式
            nbdram_time = 0;
            for (int j = 0; j < cache_count; j++) {
                in_dcacheline = inp_global_addr;
                tlm::tlm_generic_payload trans;
                trans.set_address(in_dcacheline);
                trans.set_data_length(cache_lines / 8);
                trans.set_streaming_width(cache_lines / 8);
                trans.set_command(tlm::TLM_READ_COMMAND);
#if DUMMY == 1
                trans.set_data_ptr(
                    reinterpret_cast<unsigned char *>((void *)0));
#else
                trans.set_data_ptr(
                    reinterpret_cast<unsigned char *>(dram_start));
#endif
                trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
                sc_time delay = sc_time(0, SC_NS);
                wc->isocket->b_transport(trans, delay);
                u_int64_t timer = delay.to_seconds() * 1e9;
                nbdram_time += timer;
                inp_global_addr += cache_lines;
            }

            // 计算 SRAM 写入时间
            sram_time = 0;
            for (int i = 0; i < single_read_count; i++) {
                sram_time += RAM_WRITE_LATENCY;
#if USE_SRAM_MANAGER == 1
                if (use_manager == true && i != single_read_count - 1) {
                    sram_addr_temp =
                        sram_manager_->get_address_with_offset(
                            alloc_id, sram_addr_temp * sram_bitw / 8,
                            sram_bitw / 8) /
                        (sram_bitw / 8);
                } else if (i != single_read_count - 1) {
                    sram_addr_temp = sram_addr_temp + 1;
                }
#else
                if (i != single_read_count - 1) {
                    sram_addr_temp = sram_addr_temp + 1;
                }
#endif
            }

            // 等待 DRAM 和 SRAM 操作完成
            wait(nbdram_time, SC_NS);
            if (nbdram_time < sram_time) {
                wait(sram_time - nbdram_time, SC_NS);
            }

            LOG_DEBUG(MEMORY_DEBUG)
                << "Core " << context.cid
                << " TLM single_read nbdram time: " << nbdram_time
                << " single_read_count: " << single_read_count
                << " sram_time: " << sram_time;
#endif // USE_NB_DRAMSYS
        }
    }
}


// no need to revise context.sram_addr value

void sram_spill_back_generic(TaskCoreContext &context, int data_size_in_byte,
                             u_int64_t global_addr, u_int64_t &dram_time) {
    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;
    // assert(false);
    int dma_read_count = data_size_in_byte * 8 / (int)(sram_bitw * SRAM_BANKS);
    int byte_residue =
        data_size_in_byte * 8 - dma_read_count * (sram_bitw * SRAM_BANKS);
    int single_read_count = CeilingDivision(byte_residue, sram_bitw);
    u_int64_t inp_global_addr = global_addr;

#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    auto sram_addr = context.sram_addr;
    // no use
    auto sram_addr_temp = *sram_addr;
#if USE_NB_DRAMSYS == 1
    auto nb_dcache = context.nb_dcache;
#endif
    auto s_nbdram = context.s_nbdram;
    auto e_nbdram = context.e_nbdram;

    vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    for (int i = 0; i < SRAM_BANKS; i++) {
        data_tmp[i] = 0;
    }

    u_int64_t in_dcacheline = 0;
    int cache_lines = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count = SRAM_BANKS * sram_bitw / cache_lines;
    int sram_time = 0;

#if USE_GLOBAL_DRAM == 1

    inp_global_addr =
        (global_addr / 64) * 64; // 向下取整到dram 取址的整数倍，这里是32
    int end_addr = global_addr + data_size_in_byte;
    int end_global_addr = ((end_addr + 63) / 64) * 64; // 尾地址向上取整

    int aligned_data_size_in_byte = end_global_addr - inp_global_addr;


    auto gpunb_dcache_if = context.gpunb_dcache_if;

    s_nbdram = context.start_nb_gpu_dram_event;
    e_nbdram = context.end_nb_gpu_dram_event;

    u_int64_t in_dcacheline_g = 0;
    int cache_lines_g = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count_g =
        CeilingDivision(aligned_data_size_in_byte * 8, cache_lines_g);
#endif


    sc_time start_first_write_time = sc_time_stamp();


#if USE_NB_DRAMSYS == 1
#if DRAM_BURST_BYTE > 0
    assert(DRAM_BURST_BYTE > context.defaultDataLength);
#endif

#if USE_GLOBAL_DRAM == 1
    // assert(inp_global_addr!=0);
    gpunb_dcache_if->reconfigure(inp_global_addr, cache_count_g, cache_lines_g,
                                 0);
    sc_time start_nbdram = sc_time_stamp();
    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "read_gpu", "B",
                                    Trace_event_util("read_gpu"));

    wait(*e_nbdram);
    sc_time end_nbdram = sc_time_stamp();
    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "read_gpu", "E",
                                    Trace_event_util("read_gpu"));

#else
    if (!SPEC_USE_BEHA_DRAM) {
        nb_dcache->reconfigure(inp_global_addr, dma_read_count, cache_count,
                               cache_lines, 0);
    }
    sc_time start_nbdram = sc_time_stamp();
    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " start spill back data from SRAM to DRAM";

    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "W_Dram", "B", Trace_event_util("W_Dram"));
    if (!SPEC_USE_BEHA_DRAM) {
        wait(*e_nbdram);
    } else {
        auto require_byte = dma_read_count * cache_count * cache_lines / 8;
        float need_NS = (float)require_byte / HW_BEHA_DRAM_UTIL /
                        (15.0 * GetCoreHWConfig(context.cid)->dram_bw / 8);
        int need_cycles = need_NS;
        wait(need_cycles, SC_NS);
    }
    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "W_Dram", "E", Trace_event_util("W_Dram"));
    sc_time end_nbdram = sc_time_stamp();
    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " end spill back data from SRAM to DRAM";
#endif
    u_int64_t nbdram_time = (end_nbdram - start_nbdram).to_seconds() * 1e9;

    for (int i = 0; i < dma_read_count; i++) {
        // sc_time elapsed_time;
        // hmau->mem_read_port->multiport_read(sram_addr_temp, data_tmp,
        //                                     elapsed_time);
        // u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
        // sram_time += sram_timer;
        sram_time += RAM_READ_LATENCY;
        sram_addr_temp = sram_addr_temp + SRAM_BANKS;
    }

    if (nbdram_time < sram_time) {
        wait(sram_time - nbdram_time, SC_NS);
    }

#else
    u_int64_t nbdram_time = 0;
    for (int i = 0; i < dma_read_count; i++) {
        for (int j = 0; j < cache_count; j++) {
            in_dcacheline = inp_global_addr;
            tlm::tlm_generic_payload trans;
            trans.set_address(in_dcacheline);
            trans.set_data_length(cache_lines / 8);
            trans.set_streaming_width(cache_lines / 8);
            trans.set_command(tlm::TLM_READ_COMMAND);
#if DUMMY == 1
            trans.set_data_ptr(reinterpret_cast<unsigned char *>((void *)0));
#else
            trans.set_data_ptr(reinterpret_cast<unsigned char *>(dram_start));
#endif
            trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
            sc_time delay = sc_time(0, SC_NS);
            wc->isocket->b_transport(trans, delay);
            u_int64_t timer = delay.to_seconds() * 1e9;
            nbdram_time += timer;
            // wait(delay);

            inp_global_addr += cache_lines;
        }
        sram_time = sram_time += RAM_READ_LATENCY;
        sram_addr_temp = sram_addr_temp + SRAM_BANKS;
    }
    wait(nbdram_time, SC_NS);
    if (nbdram_time < sram_time) {
        wait(sram_time - nbdram_time, SC_NS);
    }
#endif

    if (single_read_count > 0) {

#if USE_NB_DRAMSYS == 1
#if USE_GLOBAL_DRAM == 0
        nb_dcache->reconfigure(inp_global_addr +
                                   cache_lines * cache_count * dma_read_count,
                               1, cache_count, cache_lines, 0);
        start_nbdram = sc_time_stamp();

        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "W_Dram", "B",
                                        Trace_event_util("W_Dram"));
        wait(*e_nbdram);
        context.event_engine->add_event("Core " + ToHexString(context.cid),
                                        "W_Dram", "E",
                                        Trace_event_util("W_Dram"));
        end_nbdram = sc_time_stamp();

        nbdram_time = (end_nbdram - start_nbdram).to_seconds() * 1e9;
        sram_time = 0;
        // sc_bv<SRAM_BITWIDTH> data_tmp2;
        // data_tmp2 = 0;
        // sc_time elapsed_time;
        for (int i = 0; i < single_read_count; i++) {
            // mau->mem_read_port->read(sram_addr_temp, data_tmp2,
            // elapsed_time); u_int64_t sram_timer = elapsed_time.to_seconds() *
            // 1e9; sram_time += sram_timer;
            sram_addr_temp = sram_addr_temp + 1;

            sram_time += RAM_READ_LATENCY;
        }

        if (nbdram_time < sram_time) {
            wait(sram_time - nbdram_time, SC_NS);
        }
#endif
#else

        nbdram_time = 0;
        for (int j = 0; j < cache_count; j++) {
            in_dcacheline = inp_global_addr;
            tlm::tlm_generic_payload trans;
            trans.set_address(in_dcacheline);
            trans.set_data_length(cache_lines / 8);
            trans.set_streaming_width(cache_lines / 8);
            trans.set_command(tlm::TLM_READ_COMMAND);
#if DUMMY == 1
            trans.set_data_ptr(reinterpret_cast<unsigned char *>((void *)0));
#else
            trans.set_data_ptr(reinterpret_cast<unsigned char *>(dram_start));
#endif
            trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
            sc_time delay = sc_time(0, SC_NS);
            wc->isocket->b_transport(trans, delay);
            u_int64_t timer = delay.to_seconds() * 1e9;
            // dram_time += timer;
            // wait(delay);
            nbdram_time += timer;
            inp_global_addr += cache_lines;
        }

        sram_time = 0;
        for (int i = 0; i < single_read_count; i++) {
            // mau->mem_write_port->write(sram_addr_temp, data_tmp2,
            //                            elapsed_time);
            // u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
            // dram_time += sram_timer;
            sram_time += RAM_READ_LATENCY;
            sram_addr_temp = sram_addr_temp + 1;
        }
        wait(nbdram_time, SC_NS);
        if (nbdram_time < sram_time) {
            wait(sram_time - nbdram_time, SC_NS);
        }
#endif
    }


    sc_time end_first_write_time = sc_time_stamp();
    dram_time +=
        (end_first_write_time - start_first_write_time).to_seconds() * 1e9;
}


// no revise context.sram_addr value

void sram_read_generic(TaskCoreContext &context, int data_size_in_byte,
                       int sram_addr_offset, u_int64_t &dram_time,
                       AllocationID alloc_id, bool use_manager,
                       SramPosLocator *sram_pos_locator, int start_offset) {
    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;

    int dma_read_count = data_size_in_byte * 8 / (int)(sram_bitw * SRAM_BANKS);
    int bit_residue =
        data_size_in_byte * 8 - dma_read_count * (sram_bitw * SRAM_BANKS);
    int single_read_count = CeilingDivision(bit_residue, sram_bitw);

    LOG_DEBUG(MEMORY) << "Core " << context.cid << " start reading from SRAM";
    LOG_DEBUG(MEMORY) << "dma_read_count: " << dma_read_count
                      << ", single_read_count: " << single_read_count;

    auto mau = context.mau;
    auto hmau = context.hmau;
    auto sram_manager_ = context.sram_manager_;

    vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    for (int i = 0; i < SRAM_BANKS; i++) {
        data_tmp[i] = 0;
    }
#if USE_SRAM_MANAGER == 1
    if (use_manager == true) {
        sram_addr_offset = sram_manager_->get_address_index(alloc_id);
        // cout << "[INFO] sram_read_generic: alloc_id: " << alloc_id << ",
        // sram_addr_offset: " << sram_addr_offset << endl;
        sram_addr_offset =
            sram_manager_->get_address_with_offset(
                alloc_id, sram_addr_offset * sram_bitw / 8,
                start_offset / (sram_bitw / 8) * (sram_bitw / 8)) /
            (sram_bitw / 8);
        // cout << "[INFO] sram_read_generic: alloc_id: " << alloc_id << ",
        // sram_addr_offset: " << sram_addr_offset << endl;
    }

    int read_bytes = CeilingDivision(data_size_in_byte * 8, SRAM_BITWIDTH) *
                     (SRAM_BITWIDTH / 8);
    int sram_cap_bytes = sram_manager_->get_allocation_byte_capacity(alloc_id);
    // cout << "[INFO] sram_read_generic: alloc_id: " << alloc_id
    //  << ", sram_cap_bytes: " << sram_cap_bytes << ", read_bytes: " <<
    //  read_bytes << endl;
    assert(read_bytes <= sram_cap_bytes);
#endif

    int sram_time = 0;
    for (int i = 0; i < dma_read_count; i++) {
        if (i != 0) {
#if USE_SRAM_MANAGER == 1
            if (use_manager == true) {
                // std::cout << "alloc_id: " << alloc_id << std::endl;
                // sram_manager_->printAllAllocationIDs();

                sram_addr_offset =
                    sram_manager_->get_address_with_offset(
                        alloc_id, sram_addr_offset * sram_bitw / 8,
                        SRAM_BANKS * sram_bitw / 8) /
                    (sram_bitw / 8);
            } else {
                sram_addr_offset = sram_addr_offset + SRAM_BANKS;
            }
#else
            sram_addr_offset = sram_addr_offset + SRAM_BANKS;
#endif
        }

        if (!SPEC_USE_BEHA_SRAM) {
            sc_time elapsed_time;
            hmau->mem_read_port->multiport_read(sram_addr_offset, data_tmp,
                                                elapsed_time);
            u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
            dram_time += sram_timer;
        } else {
            sram_time += RAM_READ_LATENCY;
            dram_time += RAM_READ_LATENCY;
        }
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    sc_bv<SRAM_BITWIDTH> data_tmp2;
    data_tmp2 = 0;

    sc_time elapsed_time;
    sram_time = 0;
    for (int i = 0; i < single_read_count; i++) {
        if (i != 0) {
#if USE_SRAM_MANAGER == 1
            if (use_manager == true) {
                sram_addr_offset =
                    sram_manager_->get_address_with_offset(
                        alloc_id, sram_addr_offset * sram_bitw / 8,
                        sram_bitw / 8) /
                    (sram_bitw / 8);
            } else {
                sram_addr_offset = sram_addr_offset + 1;
            }
#else
            sram_addr_offset = sram_addr_offset + 1;
#endif
        }

        if (!SPEC_USE_BEHA_SRAM) {
            mau->mem_read_port->read(sram_addr_offset, data_tmp2, elapsed_time);

            u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
            dram_time += sram_timer;
        } else {
            sram_time += RAM_READ_LATENCY;
            dram_time += RAM_READ_LATENCY;
        }
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    LOG_DEBUG(MEMORY) << "Core " << context.cid << " end reading from SRAM";
}

// no revise context.sram_addr value

void sram_read_generic_temp(TaskCoreContext &context, int data_size_in_byte,
                            int sram_addr_offset, u_int64_t &dram_time) {
    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;
    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " start reading from SRAM (no pointer change)";

    int dma_read_count = data_size_in_byte * 8 / (int)(sram_bitw * SRAM_BANKS);
    int bit_residue =
        data_size_in_byte * 8 - dma_read_count * (sram_bitw * SRAM_BANKS);
    int single_read_count = CeilingDivision(bit_residue, sram_bitw);

    // cout << "[INFO] sram_read_generic: dma_read_count: " << dma_read_count <<
    // ", single_read_count: " << single_read_count << endl; cout << "[INFO]
    // sram_read_generic: data_size_in_byte: " << data_size_in_byte << ",
    // sram_addr_offset: " << sram_addr_offset << endl;

    auto mau = context.temp_mau;
    auto hmau = context.temp_hmau;

    vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    for (int i = 0; i < SRAM_BANKS; i++) {
        data_tmp[i] = 0;
    }
    int sram_time = 0;
    for (int i = 0; i < dma_read_count; i++) {
        if (!SPEC_USE_BEHA_SRAM) {
            sc_time elapsed_time;
            hmau->mem_read_port->multiport_read(sram_addr_offset, data_tmp,
                                                elapsed_time);
            u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
            dram_time += sram_timer;
        } else {
            sram_time += RAM_READ_LATENCY;
            dram_time += RAM_READ_LATENCY;
        }
        sram_addr_offset = sram_addr_offset + SRAM_BANKS;
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    sc_bv<SRAM_BITWIDTH> data_tmp2;
    data_tmp2 = 0;

    sc_time elapsed_time;
    sram_time = 0;
    for (int i = 0; i < single_read_count; i++) {
        sram_addr_offset = sram_addr_offset + 1;
        if (!SPEC_USE_BEHA_SRAM) {
            mau->mem_read_port->read(sram_addr_offset, data_tmp2, elapsed_time);

            u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
            dram_time += sram_timer;
        } else {
            sram_time += RAM_READ_LATENCY;
            dram_time += RAM_READ_LATENCY;
        }
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " end reading from SRAM (no pointer change)";
}

void sram_update_cache(TaskCoreContext &context, string label_k,
                       SramPosLocator *sram_pos_locator, int data_size_in_byte,
                       u_int64_t &dram_time, int cid) {
    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " start updating cache from SRAM";

    auto k_daddr_tmp = g_dram_kvtable[cid]->get(label_k);
    if (k_daddr_tmp.has_value()) {

    } else {
        g_dram_kvtable[cid]->add(label_k);
    }
    uint64_t k_daddr = g_dram_kvtable[cid]->get(label_k).value();


    sc_time start_first_write_time = sc_time_stamp();


    sram_pos_locator->updateKVPair(context, label_k, k_daddr,
                                   data_size_in_byte);


    sc_time end_first_write_time = sc_time_stamp();
    dram_time +=
        (end_first_write_time - start_first_write_time).to_seconds() * 1e9;

    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " end updating cache from SRAM";
}

// revise context.sram_addr value
void sram_write_append_generic(TaskCoreContext &context, int data_size_in_byte,
                               u_int64_t &dram_time, std::string label_name,
                               bool use_manager,
                               SramPosLocator *sram_pos_locator,
                               u_int64_t global_addr) {
    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " start writing to SRAM (append)";

    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;

    int dma_read_count = data_size_in_byte * 8 / (int)(sram_bitw * SRAM_BANKS);
    int byte_residue =
        data_size_in_byte * 8 - dma_read_count * (sram_bitw * SRAM_BANKS);
    int single_read_count = CeilingDivision(byte_residue, sram_bitw);

    int data_bits = data_size_in_byte * 8;
    assert((SRAM_BLOCK_SIZE * 8) % sram_bitw == 0 &&
           "SRAM_BLOCK_SIZE * 8 must be a multiple of SRAM_BITWIDTH");
    int alignment = std::max(sram_bitw, SRAM_BLOCK_SIZE * 8);

    int aligned_data_bits = static_cast<int>(std::ceil(
                                static_cast<double>(data_bits) / alignment)) *
                            alignment;
    int aligned_data_byte = aligned_data_bits / 8;
    int left_byte = aligned_data_byte - data_size_in_byte;


    sc_time start_first_write_time = sc_time_stamp();


#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.mau;
    auto hmau = context.hmau;
    auto sram_addr = context.sram_addr;
    auto sram_manager_ = context.sram_manager_;
    auto sram_addr_temp = *sram_addr;
#if USE_SRAM_MANAGER == 1
    AllocationID alloc_id = 0;
    if (use_manager == true) {
        AddrPosKey tmp;
        int flag = sram_pos_locator->findPair(label_name, tmp);
        if (flag == 0) {
            sram_pos_locator->deletePair(label_name);
        }

        SizeWAddr swa(aligned_data_byte, global_addr);

        AddrPosKey inp_key = AddrPosKey(swa);
        u_int64_t dram_time_tmp = 0;
#if ASSERT == 1
        assert(sram_pos_locator->validateTotalSize());
#endif
        sram_pos_locator->addPair(label_name, inp_key, context, dram_time_tmp,
                                  true);

        // 使用 SramManager 分配内存
        // cout << "aligned_data_byte: " << aligned_data_byte
        //      << endl;
        alloc_id = sram_manager_->allocate(aligned_data_byte);
        assert(alloc_id > 0 && "alloc_id must larger than 0 ");
        context.alloc_id_ = alloc_id;
        inp_key = AddrPosKey(context.alloc_id_, aligned_data_byte);
        inp_key.left_byte = left_byte;
        sram_pos_locator->addPair(label_name, inp_key, false);
#if ASSERT == 1
        assert(sram_pos_locator->validateTotalSize());
#endif

        if (alloc_id == 0) {
            LOG_ERROR(MEMORY) << "Failed to allocate " << aligned_data_byte
                              << " bytes from SRAM.";
        } else {
#if DEBUG_SRAM_MANAGER == 1
            LOG_DEBUG(MEMORY_DEBUG)
                << "Successfully allocated " << aligned_data_byte
                << " bytes with AllocationID: " << alloc_id;
#endif
            // 获取实际地址用于后续操作
            sram_addr_temp = sram_manager_->get_address_index(alloc_id);

            LOG_DEBUG(MEMORY_DEBUG) << "SRAM Address Index: " << sram_addr_temp;
            << std::endl;
        }
    }
#endif

    vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    for (int i = 0; i < SRAM_BANKS; i++) {
        data_tmp[i] = 0;
    }
    int sram_time = 0;

    for (int i = 0; i < dma_read_count; i++) {
        sc_time elapsed_time;
        if (i != 0) {
#if USE_SRAM_MANAGER == 1
            if (use_manager == true) {
                sram_addr_temp = sram_manager_->get_address_with_offset(
                                     alloc_id, sram_addr_temp * sram_bitw / 8,
                                     SRAM_BANKS * sram_bitw / 8) /
                                 (sram_bitw / 8);
            } else {
                sram_addr_temp = sram_addr_temp + SRAM_BANKS;
            }
#else
            sram_addr_temp = sram_addr_temp + SRAM_BANKS;
#endif
        }

        if (!SPEC_USE_BEHA_SRAM) {
            hmau->mem_read_port->multiport_write(sram_addr_temp, data_tmp,
                                                 elapsed_time);
            // u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
        } else {
            sram_time += 0; // RAM_WRITE_LATENCY;
        }
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    sram_time = 0;

    sc_bv<SRAM_BITWIDTH> data_tmp2;
    data_tmp2 = 0;

    sc_time elapsed_time;
    for (int i = 0; i < single_read_count; i++) {
        if (i != 0) {
#if USE_SRAM_MANAGER == 1
            if (use_manager == true) {
                sram_addr_temp = sram_manager_->get_address_with_offset(
                                     alloc_id, sram_addr_temp * sram_bitw / 8,
                                     sram_bitw / 8) /
                                 (sram_bitw / 8);
            } else {
                sram_addr_temp = sram_addr_temp + 1;
            }
#else
            sram_addr_temp = sram_addr_temp + 1;
#endif
        }

        if (!SPEC_USE_BEHA_SRAM) {
            mau->mem_write_port->write(sram_addr_temp, data_tmp2, elapsed_time);
        } else {
            sram_time += 0; // RAM_WRITE_LATENCY;
        }
        // u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
        // dram_time += sram_timer;
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    sc_time end_first_write_time = sc_time_stamp();
    dram_time +=
        (end_first_write_time - start_first_write_time).to_seconds() * 1e9;

    LOG_DEBUG(MEMORY) << "Core " << context.cid
                      << " end writing to SRAM (append)";

#if USE_SRAM_MANAGER == 1
    if (use_manager == true) {


    } else {
        *sram_addr = sram_addr_temp;
    }
#else
    *sram_addr = sram_addr_temp;

#endif
}
// 会修改 context.sram_addr 的数值
void sram_write_back_temp(TaskCoreContext &context, int data_size_in_byte,
                          int &temp_sram_addr, u_int64_t &dram_time) {
    int sram_bitw = GetCoreHWConfig(context.cid)->sram_bitwidth;

    int dma_read_count = data_size_in_byte * 8 / (int)(sram_bitw * SRAM_BANKS);
    int byte_residue =
        data_size_in_byte * 8 - dma_read_count * (sram_bitw * SRAM_BANKS);
    int single_read_count = CeilingDivision(byte_residue, sram_bitw);

#if USE_NB_DRAMSYS == 0
    auto wc = context.wc;
#endif
    auto mau = context.temp_mau;
    auto hmau = context.temp_hmau;

    vector<sc_bv<SRAM_BITWIDTH>> data_tmp(SRAM_BANKS);
    for (int i = 0; i < SRAM_BANKS; i++) {
        data_tmp[i] = 0;
    }
    int sram_time = 0;

    for (int i = 0; i < dma_read_count; i++) {
        if (!SPEC_USE_BEHA_SRAM) {
            sc_time elapsed_time;
            hmau->mem_read_port->multiport_write(temp_sram_addr, data_tmp,
                                                 elapsed_time);
            u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
        } else {
            sram_time += 0; // RAM_WRITE_LATENCY;
        }
        // dram_time += sram_timer;
        temp_sram_addr = temp_sram_addr + SRAM_BANKS;
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);

    sc_bv<SRAM_BITWIDTH> data_tmp2;
    data_tmp2 = 0;

    sc_time elapsed_time;
    sram_time = 0;
    for (int i = 0; i < single_read_count; i++) {
        if (!SPEC_USE_BEHA_SRAM) {
            mau->mem_write_port->write(temp_sram_addr, data_tmp2, elapsed_time);
        } else {
            sram_time += 0; // RAM_WRITE_LATENCY;
        }
        temp_sram_addr = temp_sram_addr + 1;
        // u_int64_t sram_timer = elapsed_time.to_seconds() * 1e9;
        // dram_time += sram_timer;
    }

    if (SPEC_USE_BEHA_SRAM)
        wait(sram_time, SC_NS);
}

// void check_freq(std::unordered_map<u_int64_t, u_int16_t> &freq, u_int64_t
// *tags,
//                 u_int32_t set, u_int64_t elem_tag) {
//     if (freq[elem_tag] == CACHE_MAX_FREQ) {
//         // Halve the frequency of every line in the cache set
//         u_int64_t set_base = set << CACHE_WAY_BITS;
//         for (u_int32_t i = 0; i < CACHE_WAYS; i++) {
//             u_int64_t tag = tags[set_base + i];
//             if (tag < UINT64_MAX && freq[tag] > 1)
//                 freq[tag] >>= 1;
//         }
//     }
// }

// #if DCACHE
// int dcache_replacement_policy(u_int32_t tileid, u_int64_t *tags,
//                               u_int64_t new_tag, u_int16_t &set_empty_lines)
//                               {
//     // Search the element of tags whose index into freq has the lowest value
//     int evict_dcache_idx, dcache_idx;
//     u_int16_t line_freq, min_freq = UINT16_MAX;

//     // Set base is the index of the first cache line in a set
//     // Set_id hashes the cache line tag to a set
//     // 这里感觉set在way之前，也不是不可以，低地址位先是set，然后才是对应的way
//     // bit
//     u_int32_t set = set_id_dcache(new_tag);
//     u_int64_t set_base = set << CACHE_WAY_BITS;

//     // REPLACEMENT POLICY IN HW
//     // 但是对tag进行索引的时候，还是先way 然后set
//     for (u_int32_t i = 0; i < CACHE_WAYS; i++) {
//         dcache_idx = i + set_base;
//         u_int64_t tag = tags[dcache_idx];
//         // if (tag<UINT64_MAX) line_freq = dcache_freq[tag]; //
//         // 这里表示这个dcache被是用的频率多不多
//         if (tag < UINT64_MAX)
//             line_freq = dcache_freq_v2[tag];
//         else {
//             line_freq = 0;
//             set_empty_lines++;
//         } //
//         原本的set中有一个way是空的，所以就用它，并且告诉外面原本set有空的，不用dirty

//         if (line_freq < min_freq) {
//             // Best candidate for eviction
//             evict_dcache_idx = dcache_idx;
//             min_freq = line_freq; // 找最小被是用的dcache去替换
//         }
//     }
//     // // DAHU 这个==好像有问题？？
//     // assert(evict_dcache_idx==set);
//     // cout << "\nDCACHE_DAHU " <<  evict_dcache_idx << "\n";
//     // cout << "\nDCACHE_DAHU " <<  set << "\n";
//     return evict_dcache_idx;
// }
// #endif

// u_int64_t cache_tag(u_int64_t addr) {
//     u_int64_t word_index = (u_int64_t)addr >> 2; // 4bytes in a word
//     // 全局的darray加起来，所有的tile
//     // dataset_words_per_tile dram 大小在一个tile中
//     g_data_footprint_in_words =
//         GRID_SIZE * dataset_words_per_tile; // global variable
//     word_index = word_index % g_data_footprint_in_words;
//     // 在全局darray中的索引
//     return word_index >> dcache_words_in_line_log2;
// }

// bool is_dirty(u_int64_t tag) { return dcache_dirty[tag]; }

// // LEGACY VERSION WHERE THE FUNCTION THAT ONLY RETURN THE PENALTY
// // Next-line prefetching only
// // array是全局地址，所有的tile索引的地址
// int check_dcache(int tX, int tY, u_int64_t array, u_int64_t timer,
//                  u_int64_t &time_fetched, u_int64_t &time_prefetched,
//                  u_int64_t &prefetch_tag, bool prefetch) {
//     int pu_penalty = sram_read_latency;
// #if DCACHE >= 1
// #if DCACHE == 1
//     if (dataset_cached) {
// #elif DCACHE >= 2
//     {
// #endif
//         // freq is the number of hits of a per-neighbour element
//         // // globalid 表示的是全局中tile的编号

//         // 先宽的方向后高度方向，编号
//         u_int32_t tileid = global(tX, tY);
//         // Tags that are currently in the per-tile dcache
//         u_int64_t *tags = dcache_tags[tileid];

//         // the tag is a global tag (not within the dcache)
//         // array其实按照byte的地址
//         // elem_tag 就是 全体dram地址除以dcache中一行的word数
//         u_int64_t elem_tag = cache_tag(array);

// #if DCACHE <= 4
//         // If the data I'm fetching was prefetched, then update the
//         // time_fetched, and we issue next prefetch now
//         // 如果提前一轮去拿数据，这一轮的数据上一轮拿
//         // 所以prefetch的时间是上一轮time的时间
//         // 当前的fetch时间是prefetch的时间就是上一轮的timer的时间
//         if (elem_tag == prefetch_tag && prefetch == true) {
//             time_fetched = time_prefetched;
//             time_prefetched = timer;
//             prefetch_tag++;
//         }
// #else
//         // 如果当前轮拿的话，就是当前的时间
//         time_fetched = timer;
// #endif

// #if DCACHE <= 5 // If we allow some type of data hits
//         // dcache_freq is a global array (not per tile)
//         // if (dcache_freq[elem_tag] > 0){
//         //   // ==== CACHE HIT ====
//         //   dcache_hits++;
//         //   //printf("dcache hits ++ \n");
//         // } else{
//         if (dcache_freq_v2.find(elem_tag) != dcache_freq_v2.end() &&
//             dcache_freq_v2[elem_tag] > 0) {
//             dcache_hits++;
//         } else {
// #else
//         {
// #endif
//             // ==== CACHE MISS ====
//             // printf("dcache misses ++ \n");
//             dcache_misses++;
//             // memory channel
//             u_int16_t mc_queue_id = die_id(tX, tY) * hbm_channels +
//                                     (tY * DIE_W + tX) % hbm_channels;
//             // Number of transactions that have been fetched from the HBM
//             // channel since the beggining of the program Assume that the
//             Mem.
//             // controller channel can take one request per HBM cycle
//             // 该通道历史上所有的交易总和
//             int64_t trans_count = (int64_t)mc_transactions[mc_queue_id];
//             mc_transactions[mc_queue_id] = trans_count + 1;
//             // 历史上所有的交易的总和，所以后面time_fetch的时间不能超过他
//             int64_t pu_cy_last_mc_trans =
//                 ceil_macro(trans_count * pu_mem_ratio);

//             // Fetched when the last request has satisfieda
//             // time fetched 是拿到数据的时间
//             if (pu_cy_last_mc_trans > time_fetched)
//                 time_fetched = pu_cy_last_mc_trans;

//             // May be a negative number!
//             // time 是当前运行的时间
//             int fetch_cycles_ahead = (int64_t)timer - (int64_t)time_fetched;

//             // If cycles ahead is smaller than HBM latency, then we calculate
//             // the real latency
//             int read_latency = 0;
//             int hbm_lat = (int)hbm_read_latency;
//             // fetch后还会有hbm的延迟
//             if (fetch_cycles_ahead < hbm_lat)
//                 read_latency = (hbm_lat - fetch_cycles_ahead);
//             mc_latency[mc_queue_id] += read_latency;
//             pu_penalty += read_latency;

//             u_int16_t set_empty_lines = 0;
//             int evict_dcache_idx = dcache_replacement_policy(
//                 tileid, tags, elem_tag, set_empty_lines);
// #if ASSERT_MODE && DCACHE <= 5
//             // assert(dcache_freq[elem_tag]==0);
//             assert(evict_dcache_idx < dcache_size);
// #endif
//             u_int64_t evict_tag = tags[evict_dcache_idx];

//             // An eviction occurs when dcache_freq (valid bit) is 0, but a
//             tag
//             // is already in the dcache
//             // FIXME: A local DCache can have evictions by collision of tags
//             // since the address space is not aligned locally (e.g. a dcache
//             // with 100 lines may get evictions even if its footprint is only
//             10
//             // vertices and 40 edges, since we don't consider the local
//             address
//             // space)

//             if (set_empty_lines == 0) { // IF CACHE SET FULL
//                 // Evict Line
//                 dcache_evictions++;
// #if ASSERT_MODE
//                 assert(evict_tag < UINT64_MAX);
// #endif
//                 // Mark the previous element as evicted (dcache_freq = 0)
//                 // dcache_freq[evict_tag] = 0;
//                 if (dcache_freq_v2.find(evict_tag) != dcache_freq_v2.end()) {
//                     dcache_freq_v2.erase(evict_tag);
//                 } else {
//                     assert(0);
//                 }
//                 // Writeback only if dirty!
//                 if (is_dirty(evict_tag)) {
//                     // cout << "Writeback: " << evict_tag << endl;
//                     mc_writebacks[mc_queue_id]++;
//                     mc_transactions[mc_queue_id]++;
//                 }
//             } else { // IF CACHE NOT FULL
//                      // Increase the occupancy count
// #if ASSERT_MODE
//                 assert(evict_tag == UINT64_MAX);
// #endif
//                 dcache_occupancy[tileid]++;
//             }

//             // Update the dcache tag with the new elem
//             tags[evict_dcache_idx] = elem_tag;
//             // dcache_freq[elem_tag]++;
//             if (dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end()) {
//                 dcache_freq_v2[elem_tag] = 1; // 如果不存在，则初始化为 1
//             } else {
//                 dcache_freq_v2[elem_tag]++; // 如果存在，则自增
//             }
//         }

// #if DIRECT_MAPPED == 0
//         // If the elem was not in the dcache, it's 1 now. If it was in the
//         // dcache, freq is incremented by 1. dcache_freq[elem_tag]++;
//         if (dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end()) {
//             dcache_freq_v2[elem_tag] = 1; // 如果不存在，则初始化为 1
//         } else {
//             dcache_freq_v2[elem_tag]++; // 如果存在，则自增
//         }
//         u_int32_t set = set_id_dcache(elem_tag);
//         check_freq(dcache_freq_v2, tags, set, elem_tag);
// #endif
//     }
// #endif
// #if ASSERT_MODE
//     assert(pu_penalty > 0);
// #endif
//     return pu_penalty;
// }

#if USE_L1L2_CACHE == 1
void gpu_read_generic(TaskCoreContext &context, uint64_t global_addr,
                      int data_size_in_byte, int &mem_time, bool cache_read) {

    uint64_t inp_global_addr =
        (global_addr / GPU_DRAM_ALIGNED) *
        GPU_DRAM_ALIGNED; // 向下取整到dram 取址的整数倍，这里是32
    uint64_t end_addr = global_addr + data_size_in_byte;
    uint64_t end_global_addr =
        ((end_addr + GPU_DRAM_ALIGNED - 1) / GPU_DRAM_ALIGNED) *
        GPU_DRAM_ALIGNED; // 尾地址向上取整

    uint64_t aligned_data_size_in_byte = end_global_addr - inp_global_addr;

    LOG_DEBUG(GPU) << "GPU " << context.cid << ", global_addr: " << global_addr
                   << ", inp_global_addr: " << inp_global_addr
                   << ", end_addr: " << end_addr
                   << ", end_global_addr: " << end_global_addr;
    LOG_DEBUG(GPU) << "aligned_data_size_in_byte: " << aligned_data_size_in_byte
                   << ", data_size_in_byte: " << data_size_in_byte;

    auto gpunb_dcache_if = context.gpunb_dcache_if;
    auto s_nbdram = context.start_nb_gpu_dram_event;
    auto e_nbdram = context.end_nb_gpu_dram_event;

    u_int64_t in_dcacheline = 0;
    int cache_lines = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count =
        CeilingDivision(aligned_data_size_in_byte * 8, cache_lines);


    sc_time start_first_write_time = sc_time_stamp();

    LOG_DEBUG(GPU) << "GPU " << context.cid << " read cache_count "
                   << cache_count << " cache_lines " << cache_lines;
    LOG_DEBUG(GPU) << "GPU " << context.cid << " start gpu nbdram id "
                   << gpunb_dcache_if->id;

    if (!SPEC_USE_BEHA_DRAM) {
        gpunb_dcache_if->reconfigure(inp_global_addr, cache_count, cache_lines,
                                     0);
    }

    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "read_gpu", "B",
                                    Trace_event_util("read_gpu"));
    if (!SPEC_USE_BEHA_DRAM) {
        wait(*e_nbdram);
    } else {

        auto require_byte = cache_count * cache_lines / 8;
        float need_NS = (float)require_byte / HW_BEHA_DRAM_UTIL /
                        (GPU_DRAM_BANDWIDTH)*GRID_SIZE;
        int need_cycles = need_NS;
        if (cache_read == true) {
            wait(need_cycles / 5, SC_NS);
        } else {
            wait(need_cycles, SC_NS);
        }
    }

    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "read_gpu", "E",
                                    Trace_event_util("read_gpu"));

    LOG_DEBUG(GPU) << "GPU " << context.cid << " end gpu nbdram id "
                   << gpunb_dcache_if->id;
    LOG_DEBUG(GPU) << "GPU " << context.cid << " end cache count "
                   << cache_count << " cache lines " << cache_lines;


#if USE_NB_DRAMSYS
    sc_time end_first_write_time = sc_time_stamp();
    mem_time +=
        (end_first_write_time - start_first_write_time).to_seconds() * 1e9;

    LOG_DEBUG(GPU)
        << "gpu_nbdram time: "
        << (end_first_write_time - start_first_write_time).to_string();
#endif
}

void gpu_write_generic(TaskCoreContext &context, uint64_t global_addr,
                       int data_size_in_byte, int &mem_time, bool cache_write) {

    uint64_t inp_global_addr =
        (global_addr / GPU_DRAM_ALIGNED) *
        GPU_DRAM_ALIGNED; // 向下取整到dram 取址的整数倍，这里是32
    uint64_t end_addr = global_addr + data_size_in_byte;
    uint64_t end_global_addr =
        ((end_addr + GPU_DRAM_ALIGNED - 1) / GPU_DRAM_ALIGNED) *
        GPU_DRAM_ALIGNED; // 尾地址向上取整

    uint64_t aligned_data_size_in_byte = end_global_addr - inp_global_addr;


    auto gpunb_dcache_if = context.gpunb_dcache_if;

    auto s_nbdram = context.start_nb_gpu_dram_event;
    auto e_nbdram = context.end_nb_gpu_dram_event;

    u_int64_t in_dcacheline = 0;
    int cache_lines = 1 << (dcache_words_in_line_log2 + 2 + 3);
    int cache_count =
        CeilingDivision(aligned_data_size_in_byte * 8, cache_lines);

    sc_time start_first_write_time = sc_time_stamp();

    LOG_DEBUG(GPU) << "GPU " << context.cid << " write cache_count "
                   << cache_count << " cache_lines " << cache_lines;
    LOG_DEBUG(GPU) << "GPU " << context.cid << " start gpu nbdram id "
                   << gpunb_dcache_if->id;

    if (!SPEC_USE_BEHA_DRAM) {
        gpunb_dcache_if->reconfigure(inp_global_addr, cache_count, cache_lines,
                                     1);
    }
    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "write_gpu", "B",
                                    Trace_event_util("write_gpu"));

    if (!SPEC_USE_BEHA_DRAM) {
        wait(*e_nbdram);
    } else {

        auto require_byte = cache_count * cache_lines / 8;
        float need_NS = (float)require_byte / HW_BEHA_DRAM_UTIL /
                        (GPU_DRAM_BANDWIDTH)*GRID_SIZE;
        int need_cycles = need_NS;
        if (cache_write == true) {
            wait(0, SC_NS);
        } else {
            wait(need_cycles, SC_NS);
        }
    }
    context.event_engine->add_event("Core " + ToHexString(context.cid),
                                    "write_gpu", "E",
                                    Trace_event_util("write_gpu"));

    LOG_DEBUG(GPU) << "GPU " << context.cid << " end gpu nbdram id "
                   << gpunb_dcache_if->id;


#if USE_NB_DRAMSYS
    sc_time end_first_write_time = sc_time_stamp();
    mem_time +=
        (end_first_write_time - start_first_write_time).to_seconds() * 1e9;

#endif
}
#endif

TaskCoreContext generate_context(WorkerCoreExecutor *workercore) {
    NB_GlobalMemIF *nb_global_memif = workercore->nb_global_mem_socket;
    sc_event *start_global_event = workercore->start_global_mem_event;
    sc_event *end_global_event = workercore->end_global_mem_event;

#if USE_L1L2_CACHE == 1
    // 创建类实例
    TaskCoreContext context(
        workercore->mem_access_port, workercore->high_bw_mem_access_port,
        workercore->temp_mem_access_port,
        workercore->high_bw_temp_mem_access_port, workercore->sram_addr,
        workercore->start_nb_dram_event, workercore->end_nb_dram_event,
        workercore->nb_dcache_socket, workercore->loop_cnt,
        workercore->core_context->sram_manager_,
        workercore->start_nb_gpu_dram_event, workercore->end_nb_gpu_dram_event,
        workercore->MaxDramAddr, workercore->defaultDataLength);
#elif USE_NB_DRAMSYS == 1
    TaskCoreContext context(
        workercore->mem_access_port, workercore->high_bw_mem_access_port,
        workercore->temp_mem_access_port,
        workercore->high_bw_temp_mem_access_port, workercore->sram_addr,
        workercore->start_nb_dram_event, workercore->end_nb_dram_event,
        workercore->nb_dcache_socket, workercore->sram_manager_,
        workercore->loop_cnt, workercore->MaxDramAddr,
        wokercore->defaultDataLength);
#else

    TaskCoreContext context(
        workercore->dcache_socket, workercore->mem_access_port,
        workercore->high_bw_mem_access_port, workercore->temp_mem_access_port,
        workercore->high_bw_temp_mem_access_port, workercore->sram_addr,
        workercore->start_nb_dram_event, workercore->end_nb_dram_event,
        workercore->sram_manager_, workercore->loop_cnt,
        workercore->MaxDramAddr, workercore->defaultDataLength);
#endif
    context.SetGlobalMemIF(nb_global_memif, start_global_event,
                           end_global_event);

    context.cid = workercore->cid;

#if USE_L1L2_CACHE == 1
    context.gpunb_dcache_if = workercore->gpunb_dcache_if;
#endif
    context.event_engine = workercore->event_engine;
    context.s_sram = workercore->start_sram_event;
    context.e_sram = workercore->end_sram_event;

    if (!SPEC_USE_BEHA_SRAM)
        context.sram_writer = workercore->sram_writer;

#if USE_GLOBAL_DRAM == 1
    context.event_engine = event_engine;
    context.gpunb_dcache_if = gpunb_dcache_if;
#endif
    return context;
}

uint64_t get_bank_index(uint64_t address) {
    uint64_t column_idx = address % SRAM_BANKS;
    return column_idx;
}