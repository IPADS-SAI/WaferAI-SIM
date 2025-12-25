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

// DMA Producer SystemC Module
class NB_DcacheIF : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(NB_DcacheIF);
    // TLM Initiator Socket
    tlm_utils::simple_initiator_socket<NB_DcacheIF> socket;
    sc_event *start_nb_dram_event; // 非阻塞sram访存开始标志
    sc_event *end_nb_dram_event;
    sc_event *next_dram_event;
    tlm_utils::peq_with_cb_and_phase<NB_DcacheIF> payloadEventQueue;
    sc_core::sc_time lastEndRequest = sc_core::sc_max_time();
    MemoryManager_v2 mm;

    bool transactionPostponed = false;
    bool finished = false;
    int c_id = 0;

    uint64_t transactionsSent = 0;
    uint64_t transactionsReceived = 0;
    const std::optional<unsigned int> maxPendingReadRequests;
    const std::optional<unsigned int> maxPendingWriteRequests;


    unsigned int pendingReadRequests = 0;
    unsigned int pendingWriteRequests = 0;


    // Constructor
    NB_DcacheIF(int c_id, sc_core::sc_module_name name,
                sc_event *start_nb_dram_event, sc_event *end_nb_dram_event,
                Event_engine *event_engine)
        : c_id(c_id),
          sc_module(name),
          start_nb_dram_event(start_nb_dram_event),
          end_nb_dram_event(end_nb_dram_event),
          payloadEventQueue(this, &NB_DcacheIF::peqCallback),
          mm(false),
          maxPendingReadRequests(5),
          maxPendingWriteRequests(5),
          socket("NB_Dcache_socket"),
          current_request(0),
          config_updated(false) {
        // Register the main process
        SC_THREAD(generateRequests);
        next_dram_event = new sc_event();
        socket.register_nb_transport_bw(this, &NB_DcacheIF::nb_transport_bw);
    }

    // Reconfigure the DMA producer
    void reconfigure(uint64_t base_addr, int dma_read_cnt, int cache_cnt,
                     int line_size, bool read_or_write) {
        // sc_core::sc_mutex_lock lock(config_mutex); // Protect configuration
        // variables
        base_address = base_addr;
        total_requests = dma_read_cnt * cache_cnt;
        // cache_lines = line_size;
        data_length = line_size / 8; // 假设每行按8字节分块

        LOG_DEBUG(MEMORY_DEBUG) << "Core " << c_id << " total_requests "
                                << total_requests << " line_size " << line_size;

#if DRAM_BURST_BYTE > 0
        total_requests = (total_requests * data_length + DRAM_BURST_BYTE - 1) /
                         DRAM_BURST_BYTE;
        data_length = DRAM_BURST_BYTE;
#endif
        current_request = 0;                 // Reset request counter
        config_updated = true;               // Notify the main process
        this->read_or_write = read_or_write; // 读写标志位
        (*start_nb_dram_event).notify();     // Trigger reconfiguration
    }

private:
    // Configuration variables
    uint64_t base_address; // 起始地址
    int total_requests;    // 总请求数 = dma_read_count * cache_count
    int current_request;   // 已生成请求计数
    // int cache_lines;       // 地址步进值（字节）
    int data_length;    // 传输长度单位
    bool read_or_write; // 读写标志位 0 是 读 1 是 写

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
        // 打印phase类型
        // 打印当前时间戳
#if NB_CACHE_DEBUG == 1
        std::cout << "Current time: " << sc_core::sc_time_stamp() << std::endl;
        std::cout << "Phase: ";
        if (phase == tlm::BEGIN_REQ) {
            std::cout << "BEGIN_REQ" << std::endl;
        } else if (phase == tlm::END_REQ) {
            std::cout << "END_REQ" << std::endl;
        } else if (phase == tlm::BEGIN_RESP) {
            std::cout << "BEGIN_RESP" << std::endl;
        } else if (phase == tlm::END_RESP) {
            std::cout << "END_RESP" << std::endl;
        } else {
            std::cout << "UNKNOWN" << std::endl;
        }
#endif
        if (phase == tlm::END_REQ) {
            lastEndRequest = sc_core::sc_time_stamp();

            if (nextRequestSendable())
                next_dram_event->notify();
            else
                transactionPostponed = true;
        } else if (phase == tlm::BEGIN_RESP) {
            tlm::tlm_phase nextPhase = tlm::END_RESP;
            sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
            // cout << "address!!!!!!!!!!!!!!!!!!: " << payload.get_address() <<
            // endl;
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
                << "Core " << c_id
                << " BEGIN RESP transaction sent: " << transactionsSent
                << " received: " << transactionsReceived << " finish"
                << finished;

            // If all answers were received:
            if (finished && transactionsSent == transactionsReceived) {
                finished = false;
                transactionsSent = 0;
                transactionsReceived = 0;

                LOG_DEBUG(MEMORY_DEBUG)
                    << "Core " << c_id << " Begin resp dram event";

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

            // If all answers were received:
            LOG_DEBUG(MEMORY_DEBUG)
                << "Core " << c_id
                << " End resp transaction sent: " << transactionsSent
                << " received: " << transactionsReceived << " finish"
                << finished;

            if (finished && transactionsSent == transactionsReceived) {
                finished = false;
                transactionsSent = 0;
                transactionsReceived = 0;

                LOG_DEBUG(MEMORY_DEBUG)
                    << "Core " << c_id << " End resp dram event";

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
#if NB_CACHE_DEBUG == 1
            cout << "Core " << c_id << "start generateRequests " << std::endl;
#endif
            wait(*start_nb_dram_event);

            LOG_DEBUG(MEMORY_DEBUG)
                << "Core " << c_id << " total_requests " << total_requests;

            if (total_requests > 0) {
                transactionsSent = total_requests;
                while (current_request < total_requests) {
                    // Wait for configuration to be set
                    // if (current_request >= total_requests || config_updated)
                    // {
                    //     wait(config_event); // Wait for reconfiguration
                    //     config_updated = false; // Reset the flag
                    // }
                    // 打印当前请求信息
                    // std::cout << "Request " << current_request + 1 << " of "
                    //           << total_requests << ": address=0x" << std::hex
                    //           << (base_address + current_request *
                    //           data_length)
                    //           << ", length=" << std::dec << data_length
                    //           << ", command=Read" << std::endl;
                    // std::cout << "Event: Before notified at time "
                    //           << sc_core::sc_time_stamp() << std::endl;

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
                    LOG_DEBUG(MEMORY_DEBUG)
                        << "Core " << c_id
                        << " next_dram_event notified at time "
                        << sc_core::sc_time_stamp();

                    finished = true;
                    wait(*next_dram_event);

                    // Wait for some delay (if needed)
                    // wait(sc_core::sc_time(10, sc_core::SC_NS)); // Example
                    // delay
                }
                // finished = true;
            } else {
                std::cout << "total_requests is 0, waiting for reconfiguration."
                          << std::endl;
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
#if NB_CACHE_DEBUG == 1
        cout << "send Request: " << request.address << " delay: " << delay
             << endl;
#endif
        socket->nb_transport_fw(trans, phase, delay);

        // Check response status
        // if (trans.is_response_error()) {
        //     SC_REPORT_ERROR("DMAProducer", "TLM transaction failed");
        // }
        // wait(delay);
    }
};