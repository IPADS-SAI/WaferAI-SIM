// #pragma once

// #include "systemc.h"
// #include <optional>
// #include <tlm>
// #include <tlm_utils/peq_with_cb_and_phase.h>
// #include <tlm_utils/simple_initiator_socket.h>
// #include <tlm_utils/simple_target_socket.h>

// #include "macros/macros.h"
// #include "memory/MemoryManager_v2.h"
// #include "trace/Event_engine.h"
// #include "memory/dram/utils.h"
// #include "link/chip_global_memory.h"

// class NB_GlobalMemIF : public sc_core::sc_module {
// public:
//     SC_HAS_PROCESS(NB_GlobalMemIF);
//     tlm_utils::simple_initiator_socket<NB_GlobalMemIF> socket;
//     sc_event *start_nb_dram_event;
//     sc_event *end_nb_dram_event;
//     tlm_utils::peq_with_cb_and_phase<NB_GlobalMemIF> payloadEventQueue;
//     sc_core::sc_time lastEndRequest = sc_core::sc_max_time();
//     MemoryManager_v2 mm;

//     bool transactionPostponed = false;
//     bool finished = false;

//     uint64_t transactionsSent = 0;
//     uint64_t transactionsReceived = 0;
//     const std::optional<unsigned int> maxPendingReadRequests;
//     const std::optional<unsigned int> maxPendingWriteRequests;

//     NB_GlobalMemIF(sc_core::sc_module_name name, sc_event
//     *start_nb_dram_event, sc_event *end_nb_dram_event, Event_engine
//     *event_engine)
//         : sc_module(name),
//           start_nb_dram_event(start_nb_dram_event),
//           end_nb_dram_event(end_nb_dram_event),
//           payloadEventQueue(this, &NB_GlobalMemIF::peqCallback),
//           mm(false),
//           maxPendingReadRequests(5),
//           maxPendingWriteRequests(5),
//           socket("NB_GlobalMem_socket"),
//           current_request(0),
//           config_updated(false) {
//         // Register the main process
//         SC_THREAD(generateRequests);
//         next_dram_event = new sc_event();
//         socket.register_nb_transport_bw(this,
//         &NB_GlobalMemIF::nb_transport_bw);
//     }


// };


// NB_GlobalMemIF.h
#pragma once
#include <optional>
#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include "macros/macros.h"
#include "memory/MemoryManager_v2.h"
#include "memory/dram/utils.h"
#include "trace/Event_engine.h"

//---------------------------------------------------------------------
// 用户可调整的全局宏
//---------------------------------------------------------------------
#ifndef NBGMIF_CYCLE_NS   // 时钟周期，用于对齐 BEGIN_REQ
#define NBGMIF_CYCLE_NS 1 // 1 ns = 1 GHz
#endif

//---------------------------------------------------------------------
//      NB_GlobalMemIF  ——  通用 DMA 预取 / 写回 发起器
//---------------------------------------------------------------------
class NB_GlobalMemIF : public sc_core::sc_module {
public:
    // ===== 对外接口 =========================================================
    tlm_utils::simple_initiator_socket<NB_GlobalMemIF> socket;

    /** 由外部触发本轮传输 */
    sc_core::sc_event *start_evt;
    /** 本轮全部事务完全结束时触发 */
    sc_core::sc_event *done_evt;

    // ===== 构造 / 进程注册 ===================================================
    SC_HAS_PROCESS(NB_GlobalMemIF);
    NB_GlobalMemIF(sc_core::sc_module_name name, sc_core::sc_event *start_event,
                   sc_core::sc_event *done_event, Event_engine *event_engine)
        : sc_module(name),
          socket("NB_GlobalMemIF_socket"),
          start_evt(start_event),
          done_evt(done_event),
          next_evt(new sc_core::sc_event),
          peq(this, &NB_GlobalMemIF::peq_cb),
          mm(/*allow_deallocate=*/false) {
        socket.register_nb_transport_bw(this, &NB_GlobalMemIF::nb_transport_bw);
        SC_THREAD(main_thread);
    }

    // ===== 支持的简单批量配置接口 ===========================================
    /**
     * 配置一次批量读写
     * @param base_addr     起始地址
     * @param num_reads     需要读取的行数
     * @param num_writes    紧随其后的写行数（可 0）
     * @param line_bytes    每行字节数
     * @param write_data    当写时使用的填充值（可 nullptr，内部会用 dummy）
     */
    void reconfigure(uint64_t base_addr, int num_reads, int num_writes,
                     int line_bytes,
                     const unsigned char *write_data = nullptr) {
        base_address = base_addr;
        total_reads = num_reads;
        total_writes = num_writes;
        line_size = line_bytes;
        data_template = write_data;
        reads_sent = writes_sent = reads_recv = writes_recv = 0;
        pendingReadRequests = 0;
        pendingWriteRequests = 0;
        finished_reads = (total_reads == 0);
        finished_writes = (total_writes == 0);
        (*start_evt).notify();
    }

    // -----------------------------------------------------------------
    // 下面的公有 getter 可按需添加
    // -----------------------------------------------------------------

private:
    // ======= 基本数据结构 ====================================================
    struct Request {
        enum class Cmd { READ, WRITE };
        uint64_t addr{0};
        Cmd cmd{Cmd::READ};
        unsigned int len{0};
        sc_core::sc_time delay{sc_core::SC_ZERO_TIME};
    };

    // ======= 配置变量 ========================================================
    uint64_t base_address{0};
    int total_reads{0};
    int total_writes{0};
    int line_size{64};
    const unsigned char *data_template{nullptr};

    // ======= 统计 / 状态 =====================================================
    uint64_t reads_sent{0}, writes_sent{0};
    uint64_t reads_recv{0}, writes_recv{0};

    unsigned int pendingReadRequests{0};
    unsigned int pendingWriteRequests{0};
    const std::optional<unsigned int> maxPendingReadRequests{8};
    const std::optional<unsigned int> maxPendingWriteRequests{8};

    bool finished_reads{true};
    bool finished_writes{true};

    // ======= 时序辅助 ========================================================
    sc_core::sc_event *next_evt; // 控制背压
    tlm_utils::peq_with_cb_and_phase<NB_GlobalMemIF> peq;
    MemoryManager_v2 mm;
    sc_core::sc_time last_end_req{sc_core::sc_max_time()};

    // -----------------------------------------------------------------
    // 后向路径：把所有 phase 放入 PEQ
    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload &pl,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &delay) {
        peq.notify(pl, phase, delay);
        return tlm::TLM_ACCEPTED;
    }

    // -----------------------------------------------------------------
    // PEQ 回调
    void peq_cb(tlm::tlm_generic_payload &pl, const tlm::tlm_phase &phase) {
        if (phase == tlm::END_REQ) {
            last_end_req = sc_core::sc_time_stamp();
            if (can_send_next())
                next_evt->notify();
            else
                txn_postponed = true;
        } else if (phase == tlm::BEGIN_RESP) {
            // 马上转 END_RESP
            tlm::tlm_phase next = tlm::END_RESP;
            sc_core::sc_time zero = sc_core::SC_ZERO_TIME;
            socket->nb_transport_fw(pl, next, zero);
        } else if (phase == tlm::END_RESP) {
            // 收到最终响应
            if (pl.get_command() == tlm::TLM_READ_COMMAND) {
                ++reads_recv;
                --pendingReadRequests;
            } else {
                ++writes_recv;
                --pendingWriteRequests;
            }
            pl.release();

            if (txn_postponed && can_send_next()) {
                next_evt->notify();
                txn_postponed = false;
            }
            check_batch_done();
        } else
            SC_REPORT_FATAL(name(), "Unknown phase in PEQ");
    }

    // -----------------------------------------------------------------
    // 主线程：批量产生事务
    void main_thread() {
        while (true) {
            wait(*start_evt);

            // ========= 先发 READ =========
            for (int i = 0; i < total_reads; ++i) {
                wait_for_slot();
                send_one(Request{base_address + uint64_t(i) * line_size,
                                 Request::Cmd::READ, unsigned(line_size)});
                ++reads_sent;
                ++pendingReadRequests;
            }
            finished_reads = true;
            check_batch_done(); // 可能此时就全结束

            // ========= 再发 WRITE =========
            for (int j = 0; j < total_writes; ++j) {
                wait_for_slot();
                send_one(Request{base_address +
                                     uint64_t(total_reads + j) * line_size,
                                 Request::Cmd::WRITE, unsigned(line_size)});
                ++writes_sent;
                ++pendingWriteRequests;
            }
            finished_writes = true;
            check_batch_done();
        }
    }

    // -----------------------------------------------------------------
    // 发送一条 TLM 事务
    void send_one(const Request &req) {
        tlm::tlm_generic_payload &gp = mm.allocate(req.len);
        gp.acquire();

        gp.set_address(req.addr);
        gp.set_data_length(req.len);
        gp.set_streaming_width(req.len);
        gp.set_byte_enable_length(0);
        gp.set_dmi_allowed(false);
        gp.set_command(req.cmd == Request::Cmd::READ ? tlm::TLM_READ_COMMAND
                                                     : tlm::TLM_WRITE_COMMAND);
        gp.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

        // 写操作需要一个合法 data_ptr；简单起见用模板或 dummy
        static unsigned char dummy[64]{};
        gp.set_data_ptr(const_cast<unsigned char *>(
            req.cmd == Request::Cmd::WRITE
                ? (data_template ? data_template : dummy)
                : nullptr));

        // === 对齐到时钟边界，避免冲突 ===
        sc_core::sc_time clk =
            sc_core::sc_time(NBGMIF_CYCLE_NS, sc_core::SC_NS);
        sc_core::sc_time send_t = sc_core::sc_time_stamp();
        bool misaligned = (send_t % clk) != sc_core::SC_ZERO_TIME;
        if (misaligned) {
            send_t += clk;
            send_t -= send_t % clk;
            wait(send_t - sc_core::sc_time_stamp());
        }
        if (send_t == last_end_req) { // 与上一条 END_REQ 同周期
            send_t += clk;
            wait(clk);
        }
        sc_core::sc_time delay = send_t - sc_core::sc_time_stamp();

        tlm::tlm_phase ph = tlm::BEGIN_REQ;
        socket->nb_transport_fw(gp, ph, delay);
    }

    // -----------------------------------------------------------------
    // 等待背压 & 配额
    void wait_for_slot() {
        if (!can_send_next())
            wait(*next_evt);
    }

    bool can_send_next() const {
        if (this->maxPendingReadRequests &&
            pendingReadRequests >= *this->maxPendingReadRequests)
            return false;
        if (this->maxPendingWriteRequests &&
            pendingWriteRequests >= *this->maxPendingWriteRequests)
            return false;
        return true;
    }

    // -----------------------------------------------------------------
    // 判断一批是否结束
    void check_batch_done() {
        if (finished_reads && finished_writes && reads_sent == reads_recv &&
            writes_sent == writes_recv) {
            finished_reads = finished_writes = false; // 重置
            (*done_evt).notify();
        }
    }

    // -----------------------------------------------------------------
    // 额外状态
    bool txn_postponed{false};
};
