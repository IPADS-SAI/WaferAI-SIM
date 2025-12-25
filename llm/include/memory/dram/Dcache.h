#pragma once
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <cmath>
#include <iostream>
#include <systemc>
#include <tlm>
#include <vector>

#include "defs/const.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include "memory/dramsys_wrapper.h"
#include "trace/Event_engine.h"
#include "utils/system_utils.h"

using namespace sc_core;
using namespace tlm;
using namespace std;

// 缓存模块
class DCache : public sc_module {
public:
    // TLM Socket
    tlm_utils::simple_target_socket<DCache> socket;

    gem5::memory::DRAMSysWrapper *dramSysWrapper;


    ::DRAMSys::Config::Configuration testConfig;
    tlm_utils::simple_initiator_socket<DCache> initiatorSocket;

    u_int64_t time_fetched = 0;
    u_int64_t time_prefetched = 0;
    u_int64_t prefetch_tag = 0;
    bool prefetch = false;
    int tX;
    int tY;
    int cid;

    SC_HAS_PROCESS(DCache);
    DCache(const sc_module_name &n, int cid, int idX, int idY,
           Event_engine *event_engine, std::string_view configuration,
           std::string_view resource_directory)
        : initiatorSocket("initiatorSocket"),
          testConfig(
              ::DRAMSys::Config::from_path(configuration, resource_directory)) {
        dramSysWrapper = new gem5::memory::DRAMSysWrapper("DRAMSysWrapper",
                                                          testConfig, false);
        initiatorSocket.bind(dramSysWrapper->tSocket);
        cid = cid;
        tX = idX;
        tY = idY;
        socket.register_b_transport(this, &DCache::b_transport);
        socket.register_nb_transport_fw(this, &DCache::nb_transport_fw);
        initiatorSocket.register_nb_transport_bw(this,
                                                 &DCache::nb_transport_bw);
    }
    ~DCache() {}
    void check_freq(std::unordered_map<u_int64_t, u_int16_t> &freq,
                    u_int64_t *tags, u_int32_t set, u_int64_t elem_tag) {
        if (freq[elem_tag] == CACHE_MAX_FREQ) {
            // Halve the frequency of every line in the cache set
            u_int64_t set_base = set << CACHE_WAY_BITS;
            for (u_int32_t i = 0; i < CACHE_WAYS; i++) {
                u_int64_t tag = tags[set_base + i];
                if (tag < UINT64_MAX && freq[tag] > 1)
                    freq[tag] >>= 1;
            }
        }
    }

#if DCACHE
    int dcache_replacement_policy(u_int32_t tileid, u_int64_t *tags,
                                  u_int64_t new_tag,
                                  u_int16_t &set_empty_lines) {
        // Search the element of tags whose index into freq has the lowest value
        int evict_dcache_idx, dcache_idx;
        u_int16_t line_freq, min_freq = UINT16_MAX;

        // Set base is the index of the first cache line in a set
        // Set_id hashes the cache line tag to a set
        // haddr << way << set << laddr
        u_int32_t set = set_id_dcache(new_tag);
        u_int64_t set_base = set << CACHE_WAY_BITS;

        // REPLACEMENT POLICY IN HW
        // index of tag haddr << set << way << laddr
        for (u_int32_t i = 0; i < CACHE_WAYS; i++) {
            dcache_idx = i + set_base;
            u_int64_t tag = tags[dcache_idx];
            // if (tag<UINT64_MAX) line_freq = dcache_freq[tag]; //
            if (tag < UINT64_MAX)
                line_freq = dcache_freq_v2[tag];
            else {
                line_freq = 0;
                // one way in this set is empty. do not evict
                set_empty_lines++;
            }

            if (line_freq < min_freq) {
                // Best candidate for eviction
                evict_dcache_idx = dcache_idx;
                min_freq = line_freq;
            }
        }
        // // DAHU 这个==好像有问题？？
        // assert(evict_dcache_idx==set);
        // cout << "\nDCACHE_DAHU " <<  evict_dcache_idx << "\n";
        // cout << "\nDCACHE_DAHU " <<  set << "\n";
        return evict_dcache_idx;
    }


    u_int64_t cache_tag(u_int64_t addr) {
        u_int64_t word_index = (u_int64_t)addr >> 2; // 4bytes in a word
        // dataset_words_per_tile per tile dram size in words
        g_data_footprint_in_words = dataset_words_per_tile;
        // GRID_SIZE * dataset_words_per_tile; // global variable
        word_index = word_index % g_data_footprint_in_words;
        // 在全局darray中的索引
        return word_index >> dcache_words_in_line_log2;
    }

    // bool is_dirty(u_int64_t tag) { return dcache_dirty[cid][tag]; }
    void mark_line_dirty(int tile_id, u_int64_t tag) {
        dcache_dirty[cid].insert(tag);
    }

    void clear_line_dirty(int tile_id, u_int64_t tag) {
        dcache_dirty[cid].erase(tag);
    }

    bool is_dirty(u_int64_t tag) {
        return dcache_dirty[cid].find(tag) != dcache_dirty[cid].end();
    }
#endif
    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {}

    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload &payload,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &bwDelay) {
        socket->nb_transport_bw(payload, phase, bwDelay);
        return tlm::TLM_ACCEPTED;
    }

    // 处理传输的回调函数
    void b_transport(tlm_generic_payload &trans, sc_time &delay) {
        // std::cout << "DCache: " << sc_time_stamp() << " " <<
        // trans.get_command()
        // << " " << trans.get_address() << std::endl;
        tlm_command cmd = trans.get_command();
        u_int64_t array = trans.get_address();

        sc_time current_time = sc_time_stamp();
        u_int64_t timer = current_time.to_seconds() * 1e9;

        int pu_penalty = 0;
#if DCACHE == 0
        int hbm_lat = (int)hbm_read_latency;
        int read_latency = hbm_lat;

        trans.set_response_status(tlm::TLM_OK_RESPONSE);
        pu_penalty += read_latency;
#endif

#if DCACHE == 1
        if (dataset_cached) {
            // freq is the number of hits of a per-neighbour element
            // // globalid 表示的是全局中tile的编号

            // 先宽的方向后高度方向，编号
            u_int32_t tileid = global(tX, tY);
            // Tags that are currently in the per-tile dcache
            u_int64_t *tags = dcache_tags[tileid];

            // the tag is a global tag (not within the dcache)
            // array其实按照byte的地址
            // elem_tag 就是 全体dram地址除以dcache中一行的word数
            u_int64_t elem_tag = cache_tag(array);

            if (dcache_freq_v2.find(elem_tag) != dcache_freq_v2.end() &&
                dcache_freq_v2[elem_tag] > 0) {
                dcache_hits++;
            } else {

                // ==== CACHE MISS ====
                // printf("dcache misses ++ \n");
                dcache_misses++;
                int read_latency = 0;
                u_int16_t mc_queue_id = die_id(tX, tY) * hbm_channels +
                                        (tY * DIE_W + tX) % hbm_channels;
                if (!SPEC_USE_DRAMSYS) {
                    int hbm_lat = (int)hbm_read_latency;
                    read_latency = hbm_lat;

                    trans.set_response_status(tlm::TLM_OK_RESPONSE);
                } else {
                    // Use DramSys
                    // SC_REPORT_INFO("DRAMSYS", "USE DRAMSYS");
                    sc_core::sc_time delay_dramsys = sc_core::SC_ZERO_TIME;
                    // DAHU CACHELINE 256b
                    trans.set_data_length(32);
                    trans.set_streaming_width(32);
                    initiatorSocket->b_transport(trans, delay_dramsys);
                    if (trans.get_response_status() == tlm::TLM_OK_RESPONSE) {
                        // SC_REPORT_INFO("Test", "Write request completed
                        // successfully.");
                    } else {
                        SC_REPORT_ERROR("Test", "Write/Read request failed.");
                    }
                    read_latency = delay_dramsys.to_seconds() * 1e9;
                    // read_latency = 0;
                    // std::cout << "read_latency: " << read_latency <<
                    // std::endl;
                }
                // DAHU TODO Accuracy latency ??? // Use DramSys
                // anno
                mc_latency[mc_queue_id] += read_latency;
                pu_penalty += read_latency;

                u_int16_t set_empty_lines = 0;
                int evict_dcache_idx = dcache_replacement_policy(
                    tileid, tags, elem_tag, set_empty_lines);
#if ASSERT_MODE && DCACHE <= 5
                assert(dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end());
                assert(evict_dcache_idx < dcache_size);
#endif
                u_int64_t evict_tag = tags[evict_dcache_idx];

                if (set_empty_lines == 0) { // IF CACHE SET FULL
                    // Evict Line
                    dcache_evictions++;
#if ASSERT_MODE
                    assert(evict_tag < UINT64_MAX);
#endif
                    // Mark the previous element as evicted (dcache_freq = 0)
                    // dcache_freq[evict_tag] = 0;
                    if (dcache_freq_v2.find(evict_tag) !=
                        dcache_freq_v2.end()) {
                        dcache_freq_v2.erase(evict_tag);
                    } else {
                        assert(0);
                    }
                    // Writeback only if dirty!
                    // DAHU tmp not use
                    if (is_dirty(evict_tag)) {
                        // cout << "Writeback: " << evict_tag << endl;
                        mc_writebacks[mc_queue_id]++;
                        mc_transactions[mc_queue_id]++;
                    }
                } else { // IF CACHE NOT FULL
// Increase the occupancy count
#if ASSERT_MODE
                    assert(evict_tag == UINT64_MAX);
#endif
                    dcache_occupancy[tileid]++;
                }

                // Update the dcache tag with the new elem
                tags[evict_dcache_idx] = elem_tag;
                // dcache_freq[elem_tag]++;
                if (dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end()) {
                    dcache_freq_v2[elem_tag] = 1; // 如果不存在，则初始化为 1
                } else {
                    dcache_freq_v2[elem_tag]++; // 如果存在，则自增
                }
            }

#if DIRECT_MAPPED == 0
            // If the elem was not in the dcache, it's 1 now. If it was in the
            // dcache, freq is incremented by 1. dcache_freq[elem_tag]++; cout
            // << sc_time_stamp() << ": Start DIRECT MAPPING \n" ;
            if (dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end()) {
                dcache_freq_v2[elem_tag] = 1; // 如果不存在，则初始化为 1
            } else {
                dcache_freq_v2[elem_tag]++; // 如果存在，则自增
            }
            u_int32_t set = set_id_dcache(elem_tag);
            check_freq(dcache_freq_v2, tags, set, elem_tag);
#endif
        }
#endif
// load(1);
#if ASSERT_MODE
        assert(pu_penalty > 0);
#endif
        // mem_wait_add(pu_penalty);

        // 响应传输

        delay = sc_time(pu_penalty, SC_NS);
        ; // 模拟延迟
        trans.set_response_status(TLM_OK_RESPONSE);
    }
    // tlm::tlm_sync_enum nb_transport_fw(tlm::tlm_generic_payload& trans,
    // tlm::tlm_phase& phase, sc_core::sc_time& delay){
    //     return initiatorSocket->nb_transport_fw(trans, phase, delay);
    // }

    tlm::tlm_sync_enum nb_transport_fw(tlm::tlm_generic_payload &trans,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay) {
        // 判断phase是否为begin_req
        if (phase != BEGIN_REQ) {
            // 如果不是begin_req,直接转发到initiatorSocket
            return initiatorSocket->nb_transport_fw(trans, phase, delay);
        } else {
            // 如果是begin_req,继续执行原有代码

            // std::cout << "DCache: " << sc_time_stamp() << " " <<
            // trans.get_command() << " " << trans.get_address() << std::endl;
            tlm_command cmd = trans.get_command();
            u_int64_t array = trans.get_address();

            // 获取当前仿真时间
            sc_time current_time = sc_time_stamp();

            // 将当前时间转换为纳秒并存储在u_int64_t变量中
            u_int64_t timer = current_time.to_seconds() * 1e9;

            int pu_penalty = 0;
#if DCACHE == 0
            // tlm_phase tPhase = END_RESP;
            // sc_time tDelay = sc_time(CYCLE, SC_NS);
            // return socket->nb_transport_bw(trans, tPhase, tDelay);
            return initiatorSocket->nb_transport_fw(trans, phase, delay);
#endif

#if DCACHE == 1
            if (dataset_cached) {
                // freq is the number of hits of a per-neighbour element
                // // globalid 表示的是全局中tile的编号

                // 先宽的方向后高度方向，编号
                u_int32_t tileid = global(tX, tY);
                // Tags that are currently in the per-tile dcache
                u_int64_t *tags = dcache_tags[tileid];

                // the tag is a global tag (not within the dcache)
                // array其实按照byte的地址
                // elem_tag 就是 全体dram地址除以dcache中一行的word数
                u_int64_t elem_tag = cache_tag(array);


                if (dcache_freq_v2.find(elem_tag) != dcache_freq_v2.end() &&
                    dcache_freq_v2[elem_tag] > 0) {
                    dcache_hits++;
                    if (SPEC_USE_DRAMSYS) {
                        tlm_phase tPhase = END_RESP;
                        sc_time tDelay = sc_time(CYCLE, SC_NS);
                        // wait(CYCLE, SC_NS);
                        // cout << "DCache: nb_transport_bw " << sc_time_stamp()
                        // << " " << trans.get_command() << " " <<
                        // trans.get_address() << endl;
#if DIRECT_MAPPED == 0
                        // If the elem was not in the dcache, it's 1 now. If it
                        // was in the dcache, freq is incremented by 1.
                        // dcache_freq[elem_tag]++; cout
                        // << sc_time_stamp() << ": Start DIRECT MAPPING \n" ;
                        if (dcache_freq_v2.find(elem_tag) ==
                            dcache_freq_v2.end()) {
                            dcache_freq_v2[elem_tag] =
                                1; // 如果不存在，则初始化为 1
                        } else {
                            dcache_freq_v2[elem_tag]++; // 如果存在，则自增
                        }
                        u_int32_t set = set_id_dcache(elem_tag);
                        check_freq(dcache_freq_v2, tags, set, elem_tag);

#endif
                        return socket->nb_transport_bw(trans, tPhase, tDelay);
                    }
                } else {
                    // ==== CACHE MISS ====
                    // printf("dcache misses ++ \n");
                    dcache_misses++;
                    int read_latency = 0;
                    u_int16_t mc_queue_id = die_id(tX, tY) * hbm_channels +
                                            (tY * DIE_W + tX) % hbm_channels;

                    // DAHU TODO Accuracy latency ??? // Use DramSys
                    // anno
                    mc_latency[mc_queue_id] += read_latency;
                    pu_penalty += read_latency;

                    u_int16_t set_empty_lines = 0;
                    int evict_dcache_idx = dcache_replacement_policy(
                        tileid, tags, elem_tag, set_empty_lines);
#if ASSERT_MODE && DCACHE <= 5
                    assert(dcache_freq_v2.find(elem_tag) ==
                           dcache_freq_v2.end());
                    assert(evict_dcache_idx < dcache_size);
#endif
                    u_int64_t evict_tag = tags[evict_dcache_idx];

                    // An eviction occurs when dcache_freq (valid bit) is 0, but
                    // a tag is already in the dcache
                    // FIXME: A local DCache can have evictions by collision of
                    // tags since the address space is not aligned locally (e.g.
                    // a dcache with 100 lines may get evictions even if its
                    // footprint is only 10 vertices and 40 edges, since we
                    // don't consider the local address space)

                    if (set_empty_lines == 0) { // IF CACHE SET FULL
                        // Evict Line
                        dcache_evictions++;
#if ASSERT_MODE
                        assert(evict_tag < UINT64_MAX);
#endif
                        // Mark the previous element as evicted (dcache_freq =
                        // 0) dcache_freq[evict_tag] = 0;
                        if (dcache_freq_v2.find(evict_tag) !=
                            dcache_freq_v2.end()) {
                            dcache_freq_v2.erase(evict_tag);
                        } else {
                            assert(0);
                        }
                        // Writeback only if dirty!
                        // DAHU tmp not use
                        if (is_dirty(evict_tag)) {
                            // cout << "Writeback: " << evict_tag << endl;
                            mc_writebacks[mc_queue_id]++;
                            mc_transactions[mc_queue_id]++;
                        }
                    } else { // IF CACHE NOT FULL
// Increase the occupancy count
#if ASSERT_MODE
                        assert(evict_tag == UINT64_MAX);
#endif
                        dcache_occupancy[tileid]++;
                    }

                    // Update the dcache tag with the new elem
                    tags[evict_dcache_idx] = elem_tag;
                    // dcache_freq[elem_tag]++;
                    if (dcache_freq_v2.find(elem_tag) == dcache_freq_v2.end()) {
                        dcache_freq_v2[elem_tag] =
                            1; // 如果不存在，则初始化为 1
                    } else {
                        dcache_freq_v2[elem_tag]++; // 如果存在，则自增
                    }

                    if (!SPEC_USE_DRAMSYS) {
                        delay = sc_time(pu_penalty, SC_NS); // 模拟延迟
                        assert(false &&
                               "can not use dramsys = false in ND DRAM");
                    } else {
                        sc_core::sc_time delay_dramsys = sc_core::SC_ZERO_TIME;
                        // DAHU CACHELINE 256b
#if DIRECT_MAPPED == 0
                        if (dcache_freq_v2.find(elem_tag) ==
                            dcache_freq_v2.end()) {
                            dcache_freq_v2[elem_tag] =
                                1; // 如果不存在，则初始化为 1
                        } else {
                            dcache_freq_v2[elem_tag]++; // 如果存在，则自增
                        }
                        u_int32_t set = set_id_dcache(elem_tag);
                        check_freq(dcache_freq_v2, tags, set, elem_tag);

#endif
                        return initiatorSocket->nb_transport_fw(trans, phase,
                                                                delay);
                    }
                }
            }
#endif
        }
    }

private:
};
