#pragma once
#include "systemc.h"
#include <optional>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include "defs/global.h"
#include "macros/macros.h"
#include "memory/MemoryManager_v2.h"
#include "memory/dram/utils.h"
#include "trace/Event_engine.h"
#include "utils/print_utils.h"


// 定义 Request 结构体


// DMA Producer SystemC Module
class GPUNB_dcacheIF : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(GPUNB_dcacheIF);
    // TLM Initiator Socket
    tlm_utils::simple_initiator_socket<GPUNB_dcacheIF> socket;
    sc_event *start_nb_dram_event; // 非阻塞sram访存开始标志
    sc_event *end_nb_dram_event;
    sc_event *next_dram_event;
    tlm_utils::peq_with_cb_and_phase<GPUNB_dcacheIF> payloadEventQueue;
    sc_core::sc_time lastEndRequest = sc_core::sc_max_time();
    MemoryManager_v2 mm;
    int id;
    bool transactionPostponed = false;
    bool finished = false;

    uint64_t transactionsSent = 0;
    uint64_t transactionsReceived = 0;
    const std::optional<unsigned int> maxPendingReadRequests;
    const std::optional<unsigned int> maxPendingWriteRequests;


    unsigned int pendingReadRequests = 0;
    unsigned int pendingWriteRequests = 0;


    // Constructor
    GPUNB_dcacheIF(sc_core::sc_module_name name, int id,
                   sc_event *start_nb_dram_event, sc_event *end_nb_dram_event,
                   Event_engine *event_engine)
        : sc_module(name),
          id(id),
          start_nb_dram_event(start_nb_dram_event),
          end_nb_dram_event(end_nb_dram_event),
          payloadEventQueue(this, &GPUNB_dcacheIF::peqCallback),
          mm(false),
          maxPendingReadRequests(5),
          maxPendingWriteRequests(5),
          socket("GPUNB_Dcache_socket"),
          current_request(0),
          config_updated(false) {
        // Register the main process
        SC_THREAD(generateRequests);
        next_dram_event = new sc_event();
        socket.register_nb_transport_bw(this, &GPUNB_dcacheIF::nb_transport_bw);
    }

    // Reconfigure the DMA producer
    void reconfigure(uint64_t base_addr, int cache_cnt, int line_size,
                     bool read_or_write) {
        // sc_core::sc_mutex_lock lock(config_mutex); // Protect configuration
        // variables
        base_address = base_addr;
        total_requests = cache_cnt;
        data_length = line_size / 8; // 假设每行按8字节分块
#if DRAM_BURST_BYTE > 0
        total_requests = (total_requests * data_length + DRAM_BURST_BYTE - 1) /
                         DRAM_BURST_BYTE;
        data_length = DRAM_BURST_BYTE;
        assert(data_length > 0);
#endif
        current_request = 0;                 // Reset request counter
        config_updated = true;               // Notify the main process
        this->read_or_write = read_or_write; // 读写标志位 0 是 读 1 是 写
        (*start_nb_dram_event).notify();     // Trigger reconfiguration
    }

private:
    // Configuration variables
    uint64_t base_address; // 起始地址
    int total_requests;    // 总请求数 = dma_read_count * cache_count
    int current_request;   // 已生成请求计数
    // int cache_lines;       // 地址步进值（字节）
    bool read_or_write; // 读写标志位 0 是 读 1 是 写
    int data_length;    // 传输长度单位

    // Synchronization
    sc_core::sc_event config_event; // Event to notify reconfiguration
    sc_core::sc_mutex config_mutex; // Mutex to protect configuration
    bool config_updated;            // Flag to indicate configuration update

    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload &payload,
                                       tlm::tlm_phase &phase,
                                       sc_core::sc_time &bwDelay) {
        payloadEventQueue.notify(payload, phase, bwDelay);
        return tlm::TLM_ACCEPTED;
    }

    bool nextRequestSendable() const {
        // If either the maxPendingReadRequests or maxPendingWriteRequests
        // limit is reached, do not send next payload.
        if (maxPendingReadRequests.has_value() &&
            pendingReadRequests >= maxPendingReadRequests.value())
            return false;

        if (maxPendingWriteRequests.has_value() &&
            pendingWriteRequests >= maxPendingWriteRequests.value())
            return false;

        return true;
    }

    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {

        if (phase == tlm::END_REQ) {
            lastEndRequest = sc_core::sc_time_stamp();

            if (nextRequestSendable())
                next_dram_event->notify();
            else
                transactionPostponed = true;
        } else if (phase == tlm::BEGIN_RESP) {
            tlm::tlm_phase nextPhase = tlm::END_RESP;
            sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
            socket->nb_transport_fw(payload, nextPhase, delay);

            payload.release();

            // transactionFinished();

            transactionsReceived++;

            if (payload.get_command() == tlm::TLM_READ_COMMAND)
                pendingReadRequests--;
            else if (payload.get_command() == tlm::TLM_WRITE_COMMAND)
                pendingWriteRequests--;

            // If the initiator wasn't able to send the next payload in the
            // END_REQ phase, do it now.
            if (transactionPostponed && nextRequestSendable()) {
                next_dram_event->notify();
                transactionPostponed = false;
            }

            LOG_DEBUG(MEMORY_DEBUG)
                << "End resp finished=" << finished << " sent "
                << transactionsSent << " received " << transactionsReceived;

            // If all answers were received:
            if (finished && transactionsSent == transactionsReceived) {
                finished = false;
                transactionsSent = 0;
                transactionsReceived = 0;

                LOG_DEBUG(MEMORY_DEBUG)
                    << "Core " << id << " end event notify begin resp";

                end_nb_dram_event->notify();
            }
        } else if (phase == tlm::END_RESP) {

            payload.release();

            // transactionFinished();

            transactionsReceived++;

            if (payload.get_command() == tlm::TLM_READ_COMMAND)
                pendingReadRequests--;
            else if (payload.get_command() == tlm::TLM_WRITE_COMMAND)
                pendingWriteRequests--;

            // If the initiator wasn't able to send the next payload in the
            // END_REQ phase, do it now.
            if (nextRequestSendable()) {
                next_dram_event->notify();
            } else {
                transactionPostponed = true;
            }
            // 打印完成状态和事务计数信息
            LOG_DEBUG(MEMORY_DEBUG)
                << "End resp finished " << finished << " sent "
                << transactionsSent << " received " << transactionsReceived;

            // If all answers were received:
            if (finished && transactionsSent == transactionsReceived) {
                finished = false;
                transactionsSent = 0;
                transactionsReceived = 0;

                LOG_DEBUG(MEMORY_DEBUG)
                    << "Core " << id << " end event notify end resp";

                end_nb_dram_event->notify();
            }
        } else {
            SC_REPORT_FATAL("TrafficInitiator",
                            "PEQ was triggered with unknown phase");
        }
    }

    // Main process to generate requests
    void generateRequests() {
        while (true) {
            wait(*start_nb_dram_event);

            LOG_DEBUG(MEMORY_DEBUG)
                << "Core " << id << " total_requests  " << total_requests;

            if (total_requests > 0) {
                transactionsSent =
                    total_requests; // Set transactionsSent to total_requests
                while (current_request < total_requests) {
                    Request request;
                    request.address =
                        base_address + current_request * data_length;
                    request.command =
                        (read_or_write == 0)
                            ? Request::Command::Read
                            : Request::Command::Write; // Fixed as Read
                    request.length = data_length;
                    request.delay = sc_core::SC_ZERO_TIME;

                    // Send the request through the TLM socket
                    sendRequest(request);

                    // Increment the request counter
                    ++current_request;
                    if (request.command == Request::Command::Read)
                        pendingReadRequests++;
                    else if (request.command == Request::Command::Write)
                        pendingWriteRequests++;

                    // transactionsSent++;
                    finished = true;

                    LOG_DEBUG(MEMORY_DEBUG)
                        << "Event: next_dram_event notified at time "
                        << sc_core::sc_time_stamp() << " current_request "
                        << current_request;

                    wait(*next_dram_event);
                }

                // finished = true;
                // cout << "finished= " << finished << " GPUNB_dcacheIF[" << id
                // << "]"
                //      << " sent=" << transactionsSent
                //      << " received=" << transactionsReceived << endl;
            } else {
                end_nb_dram_event->notify();
            }
        }
    }

    // Helper function to send a request via TLM
    void sendRequest(const Request &request) {
        tlm::tlm_generic_payload &trans = mm.allocate(request.length);
        tlm::tlm_phase phase = tlm::BEGIN_REQ;
        trans.acquire();
        trans.set_address(request.address);
        trans.set_data_length(request.length);
        trans.set_byte_enable_length(0);
        trans.set_streaming_width(request.length);
        trans.set_dmi_allowed(false);
        trans.set_command(request.command == Request::Command::Read
                              ? tlm::TLM_READ_COMMAND
                              : tlm::TLM_WRITE_COMMAND);
#if DUMMY == 1
        trans.set_data_ptr(reinterpret_cast<unsigned char *>((void *)0));
#else
        trans.set_data_ptr(reinterpret_cast<unsigned char *>(dram_start));
#endif
        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

        sc_core::sc_time delay = request.delay;

        sc_core::sc_time clkPeriod;
        clkPeriod = sc_core::sc_time(CYCLE, sc_core::SC_NS);

        sc_core::sc_time sendingTime = sc_core::sc_time_stamp() + delay;

        bool needsOffset = (sendingTime % clkPeriod) != sc_core::SC_ZERO_TIME;
        if (needsOffset) {
            sendingTime += clkPeriod;
            sendingTime -= sendingTime % clkPeriod;
            wait(sendingTime - sc_core::sc_time_stamp());
        }

        if (sendingTime == lastEndRequest) {
            sendingTime += clkPeriod;
            wait(clkPeriod);
        }

        delay = sendingTime - sc_core::sc_time_stamp();
        socket->nb_transport_fw(trans, phase, delay);

        // Check response status
        // if (trans.is_response_error()) {
        //     SC_REPORT_ERROR("DMAProducer", "TLM transaction failed");
        // }
        // wait(delay);
    }
};