#pragma once
#include "defs/global.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include "memory/MemoryManager_v2.h"
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>
#include <vector>

using namespace sc_core;
using namespace tlm;
using namespace std;

#define NUML1Caches 10240

#ifndef CYCLE
#define CYCLE 2
#endif
#define DEBUG 0


/*
wb 如果不valid不需要wb
多个相同地址的mshr

四个PHASE
END RESP 在L1 BUS中

other read M->S

MSHR满了的反压，处理completed

handle中的wait

hit中的时候分别的处理，考虑bw

cache中的dirty msi

trans 中的 data ptr

hit 住的时候begin resp

每一个return的状态

如果回来是end resp,不用cpu 再发end resp

内存释放

write back trans 是 null

write back mem 也会返回，不能返回给cpu 然后要给mem end resp

两个wb 的 tag是不是要区分

释放trans

给mem是tlm accepted

*/
// 定义缓存行状态（MSI协议）
enum CacheLineState { INVALID, SHARED, MODIFIED };

// 定义总线请求类型
enum BusRequestType { READ, WRITE, INVALIDATE, WRITEBACK };

// MSHR条目
struct MSHREntry {
    uint64_t address;
    BusRequestType requestType;
    sc_time requestTime;
    tlm_generic_payload *pendingTransaction;
    bool isPending;
    bool isIssue;

    MSHREntry()
        : address(0),
          requestType(READ),
          requestTime(SC_ZERO_TIME),
          pendingTransaction(nullptr),
          isPending(false),
          isIssue(true) {}
};


enum class MemOp {
    LOAD,  // Regular load
    STORE, // Regular store
    INVALID
};

struct MemOpExtension : public tlm::tlm_extension<MemOpExtension> {
    MemOp op;

    MemOpExtension(MemOp o = MemOp::LOAD) : op(o) {}

    virtual tlm_extension_base *clone() const override {
        return new MemOpExtension(op);
    }

    virtual void copy_from(const tlm_extension_base &ext) override {
        op = static_cast<const MemOpExtension &>(ext).op;
    }
};

// 缓存行
struct CacheLine {
    uint64_t tag;
    CacheLineState state;
    vector<uint8_t> data;
    bool valid;
    bool dirty;

    CacheLine(int lineSize)
        : tag(0), state(INVALID), data(lineSize, 0), valid(false) {}
};

// 缓存集
struct CacheSet {
    vector<CacheLine> lines;

    CacheSet(int associativity, int lineSize) {
        for (int i = 0; i < associativity; i++) {
            lines.push_back(CacheLine(lineSize));
        }
    }
};

// 总线请求
struct BusRequest {
    uint64_t address;
    BusRequestType requestType;
    int sourceId;
    tlm_generic_payload *transaction;

    BusRequest(uint64_t addr, BusRequestType type, int id,
               tlm_generic_payload *trans)
        : address(addr), requestType(type), sourceId(id), transaction(trans) {}
};

// 缓存基类
class CacheBase : public sc_module {
protected:
    int cacheSize;     // 缓存大小（字节）
    int lineSize;      // 缓存行大小（字节）
    int associativity; // 相联度
    int numSets;       // 缓存组数量
    int numMSHRs;      // MSHR数量

    vector<CacheSet> sets;
    vector<MSHREntry> mshrEntries;
    MemoryManager_v2 mm;

    // 地址拆分
    void parseAddress(uint64_t address, uint64_t &tag, uint64_t &setIndex,
                      uint64_t &offset) {
        int offsetBits = log2(lineSize);
        int setIndexBits = log2(numSets);

        offset = address & ((1 << offsetBits) - 1);
        setIndex = (address >> offsetBits) & ((1 << setIndexBits) - 1);
        tag = address >> (offsetBits + setIndexBits);
    }

    // 查找可用的MSHR
    int findFreeMSHR() {
        for (int i = 0; i < numMSHRs; i++) {
            if (!mshrEntries[i].isPending) {
                return i;
            }
        }
        return -1;
    }

    // 查找特定地址的MSHR
    int findMSHRByAddress(uint64_t address) {
        for (int i = 0; i < numMSHRs; i++) {
            if (mshrEntries[i].isPending && mshrEntries[i].address == address) {
                return i;
            }
        }
        return -1;
    }

public:
    CacheBase(sc_module_name name, int cacheSize, int lineSize,
              int associativity, int numMSHRs)
        : sc_module(name),
          cacheSize(cacheSize),
          lineSize(lineSize),
          associativity(associativity),
          mm(false),
          numMSHRs(numMSHRs) {

        numSets = cacheSize / (lineSize * associativity);
        sets.resize(numSets, CacheSet(associativity, lineSize));
        mshrEntries.resize(numMSHRs);
    }
};


enum class WbOp {
    MSHR, // Regular mshr
    WB    // Write back
};

enum class WbL1Op {
    MSHR_L1, // Regular mshr
    WB_L1    // Write back
};


struct WriteBackL1Extension : public tlm::tlm_extension<WriteBackL1Extension> {
    WbL1Op op;

    WriteBackL1Extension(WbL1Op o = WbL1Op::MSHR_L1) : op(o) {}

    virtual tlm_extension_base *clone() const override {
        return new WriteBackL1Extension(op);
    }

    virtual void copy_from(const tlm_extension_base &ext) override {
        op = static_cast<const WriteBackL1Extension &>(ext).op;
    }
};

// L1缓存
class L1Cache : public CacheBase {
private:
    int cacheId;
    // std::vector<CacheBlock> blocks;  // Cache blocks
    int num_blocks;

    // Add file stream for logging
    std::ofstream logFile;

    // Add a method to log cache access
    void logCacheAccess(bool isHit, string RW, uint64_t address) {
        std::string logFileName =
            "gpu_cache/L1Cache_cid_" + std::to_string(cacheId) + ".log";

        logFile.open(logFileName, std::ios::app);
        if (logFile.is_open()) {
            logFile << "Core " << cacheId << " Rw " << RW
                    << " Time: " << sc_core::sc_time_stamp().to_string()
                    << " ns Address: 0x" << std::hex << address << std::dec
                    << " Result: " << (isHit ? "HIT" : "MISS") << std::endl;
            logFile.close();
        } else {

            assert(false);
        }
    }


    // 统一的请求结构体
    struct CacheRequest {
        enum RequestType { READ_REQ, WRITE_REQ, WRITEBACK_REQ };

        RequestType type;
        uint64_t address;
        uint8_t *data;
        int dataLength;
        tlm_generic_payload *originalTrans;

        CacheRequest(RequestType t, uint64_t addr, uint8_t *d, int len,
                     tlm_generic_payload *orig = nullptr)
            : type(t),
              address(addr),
              data(d),
              dataLength(len),
              originalTrans(orig) {}
    };

    // 添加写回请求队列和事件
    struct WritebackRequest {
        uint64_t address;
        uint8_t *data;
        int lineSize;
    };
    std::queue<WritebackRequest> writebackQueue;
    sc_event newWritebackRequest;
    queue<CacheRequest> requestQueue;
    sc_event newRequest;
    sc_event mshrevent;

public:
    tlm_utils::simple_initiator_socket<L1Cache> bus_socket;
    tlm_utils::simple_target_socket<L1Cache> cpu_socket;
    tlm_utils::peq_with_cb_and_phase<L1Cache> payloadEventQueue;

    SC_HAS_PROCESS(L1Cache);

    L1Cache(sc_module_name name, int id, int cacheSize, int lineSize,
            int associativity, int numMSHRs)
        : CacheBase(name, cacheSize, lineSize, associativity, numMSHRs),
          cacheId(id),
          payloadEventQueue(this, &L1Cache::peqCallback),
          bus_socket("bus_socket"),
          cpu_socket("cpu_socket") {
        std::string logFileName =
            "gpu_cache/L1Cache_cid_" + std::to_string(id) + ".log";
        logFile.open(logFileName, std::ios::app);

        if (!logFile.is_open()) {
            SC_REPORT_ERROR(
                "L1Cache", ("Failed to open log file: " + logFileName).c_str());
        }


        cpu_socket.register_nb_transport_fw(this, &L1Cache::nb_transport_fw);
        bus_socket.register_nb_transport_bw(this, &L1Cache::nb_transport_bw);

        SC_THREAD(processMSHRs);
        SC_THREAD(writebackHandler);    // 注册写回处理线程
        SC_THREAD(processRequestQueue); // 注册统一请求处理线程
    }
    ~L1Cache() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    // 修改writebackHandler线程，将写回请求转发到统一队列

    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {

        if (phase == tlm::END_RESP) {
            tlm_phase l2Phase = END_RESP;
            sc_time l2Delay = SC_ZERO_TIME;
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId << "]: END RESP to Processor."
                 << " Time stamp: " << sc_time_stamp() << endl;
#endif
            cpu_socket->nb_transport_bw(payload, l2Phase, l2Delay);
        } else if (phase == tlm::END_REQ) {
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId << "]: END REQ."
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << payload.get_address() << endl;
#endif

            tlm_phase l2Phase = END_REQ;
            sc_time l2Delay = SC_ZERO_TIME;

            cpu_socket->nb_transport_bw(payload, l2Phase, l2Delay);
        } else {
            assert(false);
        }
    }
    void writebackHandler() {
        while (true) {
            if (writebackQueue.empty()) {
#if DEBUG == 1
                SC_REPORT_INFO("Event Wait,(Waiting for for: ",
                               newWritebackRequest.name());
#endif

                wait(newWritebackRequest); // 等待新的写回请求
            }

            // 处理队列中的写回请求
            WritebackRequest req = writebackQueue.front();
            writebackQueue.pop();

            // 将写回请求添加到统一队列
            CacheRequest unifiedReq(CacheRequest::WRITEBACK_REQ, req.address,
                                    req.data, req.lineSize);
            requestQueue.push(unifiedReq);
            newRequest.notify(); // 通知请求处理线程
        }
    }

    // 修改processMSHRs方法，将MSHR请求转发到统一队列
    void processMSHRs() {
        while (true) {
#if DEBUG == 1
            SC_REPORT_INFO("Event Wait,(Waiting for for: ", "L1MSHR");
#endif
            // 检查并处理MSHR中的请求
            bool tmp = false;

            for (int i = 0; i < numMSHRs; i++) {
                if (mshrEntries[i].isIssue == false) {
                    tmp = true;
                    break;
                }
            }
            if (tmp == false) {
                wait(mshrevent);
            }

            for (int i = 0; i < numMSHRs; i++) {
                if (mshrEntries[i].isIssue == false) {
                    // 处理MSHR中的请求...
                    // wait(10,SC_NS);
                    mshrEntries[i].isIssue = true;
                    tlm_generic_payload *trans =
                        mshrEntries[i].pendingTransaction;
                    uint64_t addr = trans->get_address();

                    if (mshrEntries[i].requestType == READ) {
                        // 创建读请求并加入统一队列
                        uint8_t *data = new uint8_t[trans->get_data_length()];
                        // memcpy(data, trans->get_data_ptr(),
                        // trans->get_data_length());

                        CacheRequest req(CacheRequest::READ_REQ, addr, data,
                                         trans->get_data_length(), trans);
                        requestQueue.push(req);
                        newRequest.notify();

                    } else if (mshrEntries[i].requestType == WRITE) {
                        // 创建写请求并加入统一队列
                        uint8_t *data = new uint8_t[trans->get_data_length()];
                        // memcpy(data, trans->get_data_ptr(),
                        // trans->get_data_length());

                        CacheRequest req(CacheRequest::WRITE_REQ, addr, data,
                                         trans->get_data_length(), trans);
                        requestQueue.push(req);
                        newRequest.notify();
                    } else {
                        assert(false);
                    }
                    // wait(CYCLE, SC_NS); // 定期检查
                }
            }
        }
    }

    // 添加新的统一请求处理线程
    void processRequestQueue() {
        while (true) {
            if (requestQueue.empty()) {
#if DEBUG == 1
                SC_REPORT_INFO("Event Wait,(Waiting for for L1: ",
                               newRequest.name());
#endif
                wait(newRequest); // 等待新请求
            }

            // 处理队列中的请求
            CacheRequest req = requestQueue.front();
            requestQueue.pop();

            // 创建事务
            tlm_generic_payload *newTrans = req.originalTrans;

            switch (req.type) {
            case CacheRequest::READ_REQ: {
                newTrans->set_command(TLM_READ_COMMAND);
                MemOpExtension *op_ext = new MemOpExtension(MemOp::STORE);
                newTrans->set_extension(op_ext);
                WriteBackL1Extension *wb_ext =
                    new WriteBackL1Extension(WbL1Op::MSHR_L1);
                newTrans->set_extension(wb_ext);
                break;
            }

            case CacheRequest::WRITE_REQ: {

                MemOpExtension *op_ext = new MemOpExtension(MemOp::LOAD);
                newTrans->set_extension(op_ext);
                newTrans->set_command(TLM_WRITE_COMMAND);
                WriteBackL1Extension *wb_ext =
                    new WriteBackL1Extension(WbL1Op::MSHR_L1);
                newTrans->set_extension(wb_ext);
                break;
            }
            case CacheRequest::WRITEBACK_REQ: {

                tlm_generic_payload *newTrans2 = &(mm.allocate(req.dataLength));
                newTrans = newTrans2;
                newTrans->acquire();
                MemOpExtension *op_ext = new MemOpExtension(MemOp::LOAD);
                newTrans->set_extension(op_ext);
                newTrans->set_command(TLM_WRITE_COMMAND);
                WriteBackL1Extension *wb_ext =
                    new WriteBackL1Extension(WbL1Op::WB_L1);
                newTrans->set_extension(wb_ext);
                break;
            }
            }

            newTrans->set_address(req.address);
            newTrans->set_data_ptr(req.data);
            newTrans->set_data_length(req.dataLength);

            // 在下一个时钟周期发送请求
            wait(SC_ZERO_TIME); // 等待当前delta周期结束

            // 发送请求
            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = BEGIN_REQ;
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId << "]: SEND REQ to BUS."
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << req.address << endl;

#endif
            bus_socket->nb_transport_fw(*newTrans, phase, delay);

            // 等待响应（简化处理）
            wait(CYCLE, SC_NS);

            // 如果是写回请求，释放数据内存
            if (req.type == CacheRequest::WRITEBACK_REQ) {
                delete[] req.data;
            }
        }
    }

    // 处理CPU发来的请求
    tlm_sync_enum nb_transport_fw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {
        if (phase == BEGIN_REQ) {
            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);

            if (trans.get_command() == TLM_READ_COMMAND) {
                // 检查是否命中
                bool hit = false;
                for (auto &line : sets[setIndex].lines) {
                    if (line.valid && line.tag == tag &&
                        (line.state == SHARED || line.state == MODIFIED)) {
                        hit = true;
#if GPU_CACHE_DEBUG == 1
                        cout << "L1Cache [" << cacheId << "]: READ HIT."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << endl;
#endif
                        if (GPU_CACHE_LOG) {
                            logCacheAccess(true, "READ", addr);
                        }
                        delay += sc_time(CYCLE, SC_NS); // 命中延迟

                        phase = END_RESP;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        return TLM_UPDATED;
                    }
                }

                if (!hit) {
                    // 未命中，放入MSHR
#if GPU_CACHE_DEBUG == 1
                    cout << "L1Cache [" << cacheId << "]: READ MISS."
                         << " Time stamp: " << sc_time_stamp()
                         << " Address: " << addr << endl;
#endif
                    if (GPU_CACHE_LOG) {
                        logCacheAccess(false, "READ", addr);
                    }
                    int mshrIndex = findFreeMSHR();
                    if (mshrIndex >= 0) {
#if MSHRHIT == 2
                        bool mshr_hit = false;
                        uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                        for (int i = 0; i < mshrEntries.size(); i++) {
                            if (mshrEntries[i].isPending == true) {
                                parseAddress(mshrEntries[i].address, mshr_tag,
                                             mshr_setIndex, mshr_offset);
                                if (mshr_tag == tag &&
                                    mshr_setIndex == setIndex) {

                                    assert(false);
                                }
                            }
                        }


#endif
                        mshrEntries[mshrIndex].address = addr;
                        mshrEntries[mshrIndex].requestType = READ;
                        mshrEntries[mshrIndex].requestTime = sc_time_stamp();
                        mshrEntries[mshrIndex].pendingTransaction = &trans;
                        mshrEntries[mshrIndex].isPending = true;
                        mshrEntries[mshrIndex].isIssue = false;
                        phase = END_REQ;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        mshrevent.notify();

                        // 返回ACCEPTED，表示请求已接受但尚未完成
                        return TLM_ACCEPTED;
                    } else {
#if GPU_CACHE_DEBUG == 1
                        cout << "L1Cache [" << cacheId << "]: mshr full."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << trans.get_address() << endl;
#endif
                        // MSHR已满，拒绝请求
                        trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
                        return TLM_COMPLETED;
                    }
                }
            } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                // 写请求类似逻辑
                bool hit = false;
                for (auto &line : sets[setIndex].lines) {
                    if (line.valid && line.tag == tag) {
                        hit = true;
#if GPU_CACHE_DEBUG == 1
                        cout << "L1Cache [" << cacheId << "]: WRITE HIT."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << endl;
#endif
                        if (GPU_CACHE_LOG) {
                            logCacheAccess(true, "WRITE", addr);
                        }
                        if (line.state == MODIFIED) {
                            // 如果是M状态，直接写入
                            // memcpy(&line.data[0], trans.get_data_ptr(),
                            // trans.get_data_length());
                            delay += sc_time(CYCLE, SC_NS); // 命中延迟

                            phase = END_RESP;
                            sc_time bwDelay =
                                sc_core::sc_time(CYCLE, sc_core::SC_NS);
                            payloadEventQueue.notify(trans, phase, bwDelay);
                            return TLM_UPDATED;
                        } else if (line.state == SHARED) {
                            // 如果是S状态，需要升级到M状态
                            // 发送总线请求通知其他缓存


                            // 发送invalidate请求到总线
                            tlm_generic_payload *invalidateTrans =
                                &(mm.allocate(1));
                            invalidateTrans->acquire();
                            invalidateTrans->set_address(addr);
                            MemOpExtension *op_ext =
                                new MemOpExtension(MemOp::INVALID);
                            invalidateTrans->set_extension(op_ext);

                            sc_time busDelay = SC_ZERO_TIME;
                            tlm_phase busPhase = BEGIN_REQ;
                            bus_socket->nb_transport_fw(*invalidateTrans,
                                                        busPhase, busDelay);
                            phase = END_RESP;
                            sc_time bwDelay =
                                sc_core::sc_time(CYCLE, sc_core::SC_NS);
                            payloadEventQueue.notify(trans, phase, bwDelay);
                            return TLM_ACCEPTED;
                        } else {
                            assert(false);
                        }
                    }
                }

                if (!hit) {
                    // 未命中，放入MSHR
                    int mshrIndex = findFreeMSHR();
#if GPU_CACHE_DEBUG == 1
                    cout << "L1Cache [" << cacheId << "]: WRITE MISS."
                         << " Time stamp: " << sc_time_stamp()
                         << " Address: " << addr << endl;
#endif
                    if (GPU_CACHE_LOG) {
                        logCacheAccess(false, "WRITE", addr);
                    }
                    if (mshrIndex >= 0) {
#if MSHRHIT == 2
                        bool mshr_hit = false;
                        uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                        for (int i = 0; i < mshrEntries.size(); i++) {
                            if (mshrEntries[i].isPending == true) {
                                parseAddress(mshrEntries[i].address, mshr_tag,
                                             mshr_setIndex, mshr_offset);
                                if (mshr_tag == tag &&
                                    mshr_setIndex == setIndex) {

                                    assert(false);
                                }
                            }
                        }


#endif
                        mshrEntries[mshrIndex].address = addr;
                        mshrEntries[mshrIndex].requestType = WRITE;
                        mshrEntries[mshrIndex].requestTime = sc_time_stamp();
                        mshrEntries[mshrIndex].pendingTransaction = &trans;
                        mshrEntries[mshrIndex].isPending = true;
                        mshrEntries[mshrIndex].isIssue = false;

                        phase = END_REQ;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        mshrevent.notify();
                        return TLM_ACCEPTED;
                    } else {
                        trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
                        return TLM_COMPLETED;
                    }
                }
            } else {
                assert(false);
            }
        } else if (phase == END_RESP) {
            // 处理结束响应

            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = END_RESP;
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId << "]: END_RESP TO BUS."
                 << " Time stamp: " << sc_time_stamp() << endl;
#endif
            bus_socket->nb_transport_fw(trans, phase, delay);
            return TLM_COMPLETED;
        }

        return TLM_ACCEPTED;
    }

    // 处理L2/总线发来的响应
    tlm_sync_enum nb_transport_bw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {

        WriteBackL1Extension *op_ext =
            trans.get_extension<WriteBackL1Extension>();
        if (!op_ext) {
            SC_REPORT_FATAL("L1Cache", "Missing WB Type");
            return TLM_COMPLETED; // 修改为返回
        }
        if (phase == END_REQ) {
            // sc_time busDelay = SC_ZERO_TIME;
            // cpu_socket->nb_transport_bw(trans, phase, busDelay);
        } else if (phase == BEGIN_RESP && op_ext->op == WbL1Op::MSHR_L1) {
            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId
                 << "]: BEGIN RESP FROM BUS for L1 MSHR."
                 << " Time stamp: " << sc_time_stamp() << " Address: " << addr
                 << endl;
#endif

            // 查找对应的MSHR
            int mshrIndex = findMSHRByAddress(addr);
            assert(mshrIndex >= 0 && "Addr is not aligned");
            if (mshrIndex >= 0) {
                if (trans.get_command() == TLM_READ_COMMAND) {
                    // 处理读响应
                    // 为缓存行分配空间
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        // 如果要替换的行是M状态，需要写回
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            // 假设 lineSize 和 numSets 是固定的
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            // 使用预计算的值
                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);
                            // 将写回请求加入队列
                            WritebackRequest req;
                            req.address = writebackAddr;
                            req.data = new uint8_t[lineSize];
                            // memcpy(req.data,
                            // &sets[setIndex].lines[replaceIndex].data[0],
                            // lineSize);
                            req.lineSize = lineSize;

                            writebackQueue.push(req);
                            newWritebackRequest.notify();

                            // 不需要等待写回完成，可以继续处理当前请求
                        }
                    }
#if GPU_CACHE_DEBUG == 1
                    cout << "L1Cache [" << cacheId << "]: REPLACE INDEX."
                         << " Time stamp: " << sc_time_stamp()
                         << " Address: " << addr << endl;
                    cout << "setIndex " << setIndex << " replaceIndex "
                         << replaceIndex << " Tag " << tag << endl;
#endif

                    // 更新缓存行
                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = SHARED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // trans.get_data_ptr(), lineSize);

                    // 更新原始事务数据
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(origTrans->get_data_ptr(), trans.get_data_ptr(),
                    // origTrans->get_data_length());

                    // 标记MSHR为空闲
                    mshrEntries[mshrIndex].isPending = false;

                    // 回复CPU
                    tlm_phase cpuPhase = BEGIN_RESP;
                    sc_time cpuDelay = SC_ZERO_TIME;
                    cpu_socket->nb_transport_bw(*origTrans, cpuPhase, cpuDelay);
                } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                    // 处理写响应
                    // 类似读响应的逻辑
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 发起写回请求
                            // 计算写回地址
                            // 假设 lineSize 和 numSets 是固定的
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            // 使用预计算的值
                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 将写回请求加入队列
                            WritebackRequest req;
                            req.address = writebackAddr;
                            req.data = new uint8_t[lineSize];
                            // memcpy(req.data,
                            // &sets[setIndex].lines[replaceIndex].data[0],
                            // lineSize);
                            req.lineSize = lineSize;

                            writebackQueue.push(req);
                            newWritebackRequest.notify();
                        }
                    }

                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = MODIFIED;

                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // origTrans->get_data_ptr(), origTrans->get_data_length());

                    mshrEntries[mshrIndex].isPending = false;

                    tlm_phase cpuPhase = BEGIN_RESP;
                    sc_time cpuDelay = SC_ZERO_TIME;
                    cpu_socket->nb_transport_bw(*origTrans, cpuPhase, cpuDelay);
                } else {
                    assert(false);
                }
            }
        } else if (phase == END_RESP && op_ext->op == WbL1Op::MSHR_L1) {
            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);
#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId
                 << "]: END RESP FROM L2 HIT FOR L1 MSHR."
                 << " Time stamp: " << sc_time_stamp() << " Address: " << addr
                 << endl;
#endif


            // 查找对应的MSHR
            int mshrIndex = findMSHRByAddress(addr);
            if (mshrIndex >= 0) {
                if (trans.get_command() == TLM_READ_COMMAND) {
                    // 处理读响应
                    // 为缓存行分配空间
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        // 如果要替换的行是M状态，需要写回
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            // 假设 lineSize 和 numSets 是固定的
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            // 使用预计算的值
                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);
                            // 将写回请求加入队列
                            WritebackRequest req;
                            req.address = writebackAddr;
                            req.data = new uint8_t[lineSize];
                            // memcpy(req.data,
                            // &sets[setIndex].lines[replaceIndex].data[0],
                            // lineSize);
                            req.lineSize = lineSize;

                            writebackQueue.push(req);
                            newWritebackRequest.notify();

                            // 不需要等待写回完成，可以继续处理当前请求
                        }
                    }

                    // 更新缓存行
                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = SHARED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // trans.get_data_ptr(), lineSize);

                    // 更新原始事务数据
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(origTrans->get_data_ptr(), trans.get_data_ptr(),
                    // origTrans->get_data_length());

                    // 标记MSHR为空闲
                    mshrEntries[mshrIndex].isPending = false;

                    // 回复CPU
                    tlm_phase cpuPhase = END_RESP;
                    sc_time cpuDelay = SC_ZERO_TIME;
                    cpu_socket->nb_transport_bw(*origTrans, cpuPhase, cpuDelay);
                } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                    // 处理写响应
                    // 类似读响应的逻辑
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 发起写回请求
                            // 计算写回地址
                            // 假设 lineSize 和 numSets 是固定的
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            // 使用预计算的值
                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 将写回请求加入队列
                            WritebackRequest req;
                            req.address = writebackAddr;
                            req.data = new uint8_t[lineSize];
                            // memcpy(req.data,
                            // &sets[setIndex].lines[replaceIndex].data[0],
                            // lineSize);
                            req.lineSize = lineSize;

                            writebackQueue.push(req);
                            newWritebackRequest.notify();
                        }
                    }

                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = MODIFIED;

                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // origTrans->get_data_ptr(), origTrans->get_data_length());

                    mshrEntries[mshrIndex].isPending = false;

                    tlm_phase cpuPhase = END_RESP;
                    sc_time cpuDelay = SC_ZERO_TIME;
                    cpu_socket->nb_transport_bw(*origTrans, cpuPhase, cpuDelay);
                } else {
                    assert(false);
                }
            }
            return TLM_UPDATED;
        } else if (op_ext->op == WbL1Op::WB_L1) {

#if GPU_CACHE_DEBUG == 1
            cout << "L1Cache [" << cacheId << "]: END RESP L1 WB."
                 << " Time stamp: " << sc_time_stamp() << endl;
#endif


            // DAHU 释放L1 WB
        } else {
            assert(false);
        }

        return TLM_ACCEPTED;
    }

    // 处理总线发来的无效化请求（用于MSI协议）
    void handleInvalidateRequest(uint64_t address) {
        uint64_t tag, setIndex, offset;
        parseAddress(address, tag, setIndex, offset);

        for (auto &line : sets[setIndex].lines) {
            if (line.valid && line.tag == tag) {
                if (line.state == MODIFIED) {
                    // 如果是M状态，需要写回
                    // 发起写回请求
                    // 简化实现...
                    line.state = INVALID;
                    line.valid = false;
                } else if (line.state == SHARED) {
                    // 如果是S状态，直接无效化
                    line.state = INVALID;
                    line.valid = false;
                } else {
                    assert(false);
                }
            }
        }
    }
};

// 首先定义一个事务扩展类来存储源ID
class SourceIDExtension : public tlm::tlm_extension<SourceIDExtension> {
public:
    SourceIDExtension() : source_id(-1) {}

    void set_id(int id) { source_id = id; }
    int get_id() const { return source_id; }

    // 必须实现的克隆函数
    tlm::tlm_extension_base *clone() const override {
        SourceIDExtension *ext = new SourceIDExtension();
        ext->source_id = this->source_id;
        return ext;
    }

    // 必须实现的拷贝函数
    void copy_from(tlm::tlm_extension_base const &ext) override {
        source_id = static_cast<SourceIDExtension const &>(ext).source_id;
    }

private:
    int source_id;
};

// 总线模块
class Bus : public sc_module {
private:
    vector<L1Cache *> l1Caches;
    queue<BusRequest> requestQueue;
    sc_event newRequest;

public:
    tlm_utils::simple_initiator_socket<Bus> l2_socket;
    tlm_utils::simple_target_socket_tagged<Bus>
        *l1_sockets[NUML1Caches]; // 面向L1 Cache的目标socket


    SC_HAS_PROCESS(Bus);

    Bus(sc_module_name name, int numL1Caches)
        : sc_module(name), l2_socket("l2_socket") {


        for (int i = 0; i < numL1Caches; i++) {
            char txt[30];
            sprintf(txt, "l1_socket_%d", i);
            l1_sockets[i] =
                new tlm_utils::simple_target_socket_tagged<Bus>(txt);
            l1_sockets[i]->register_nb_transport_fw(this, &Bus::nb_transport_fw,
                                                    i);
        }
        l2_socket.register_nb_transport_bw(this, &Bus::nb_transport_bw);

        SC_THREAD(processRequests);
    }

    void addL1Cache(L1Cache *cache) { l1Caches.push_back(cache); }

    void processRequests() {
        while (true) {
#if GPU_CACHE_DEBUG == 1
            cout << "BUS FROM L1CACHE ["
                 << "]: processRequests "
                 << " Time stamp: " << sc_time_stamp() << endl;

#endif
            if (requestQueue.empty()) {
#if DEBUG == 1
                SC_REPORT_INFO("Event Wait,(Waiting for for Bus: ",
                               newRequest.name());
#endif
                wait(newRequest);
            }

            // 处理队列中的请求
            BusRequest request = requestQueue.front();


            if (request.requestType == READ || request.requestType == WRITE) {
                // 传递到L2
                tlm_phase phase = BEGIN_REQ;
                sc_time delay = SC_ZERO_TIME;
                tlm::tlm_sync_enum status;
                SourceIDExtension *id_ext;
                (*request.transaction).get_extension(id_ext);

                if (!id_ext) {
                    SC_REPORT_ERROR("Bus", "Missing source ID extension");
                }

                int targetId = id_ext->get_id();
#if GPU_CACHE_DEBUG == 1
                cout << "BUS FROM L1CACHE [" << targetId
                     << "]: BUS SEND REQ TO L2"
                     << " Time stamp: " << sc_time_stamp()
                     << " Address: " << request.address << endl;
#endif

                status = l2_socket->nb_transport_fw(*request.transaction, phase,
                                                    delay);
                if (status == TLM_COMPLETED) {
                    wait(CYCLE, SC_NS); // 总线仲裁延迟

                } else {
                    wait(CYCLE, SC_NS); // 总线仲裁延迟
                    requestQueue.pop();
                }


            } else if (request.requestType == INVALIDATE) {
                // 发送无效化请求到所有其他L1
                for (int i = 0; i < l1Caches.size(); i++) {
                    if (i != request.sourceId) {
                        l1Caches[i]->handleInvalidateRequest(request.address);
                    }
                }
#if GPU_CACHE_DEBUG == 1

                SourceIDExtension *id_ext;
                (*request.transaction).get_extension(id_ext);

                if (!id_ext) {
                    SC_REPORT_ERROR("Bus", "Missing source ID extension");
                }

                int targetId = id_ext->get_id();
                cout << "BUS FROM L1CACHE [" << targetId << "]: INVAlidate "
                     << " Time stamp: " << sc_time_stamp() << endl;

#endif
                wait(CYCLE, SC_NS); // 总线仲裁延迟
                requestQueue.pop();

                // 无效化请求完成后，回复发起者
                // tlm_phase phase = BEGIN_RESP;
                // sc_time delay = SC_ZERO_TIME;
                // DAHU 可以不需要？
                // l1_sockets[request.sourceId]->nb_transport_bw(*request.transaction,
                // phase, delay);
            } else {
                assert(false);
            }
        }
    }

    // 处理L1发来的请求
    tlm_sync_enum nb_transport_fw(int id, tlm_generic_payload &trans,
                                  tlm_phase &phase, sc_time &delay) {
        if (phase == BEGIN_REQ) {
            // 确定请求类型和来源
            BusRequestType reqType;
            int sourceId = id;


            // 添加源ID扩展
            SourceIDExtension *id_ext = new SourceIDExtension();
            id_ext->set_id(sourceId);
            trans.set_extension(id_ext);

            MemOpExtension *op_ext = trans.get_extension<MemOpExtension>();
            if (!op_ext) {
                SC_REPORT_FATAL("L1Cache", "Missing Op Type");
                return TLM_COMPLETED; // 修改为返回
            }

            if (op_ext->op == MemOp::LOAD) {
                reqType = READ;
            } else if (op_ext->op == MemOp::STORE) {
                reqType = WRITE;
            } else {
                reqType = INVALIDATE; // 简化处理
            }
#if GPU_CACHE_DEBUG == 1
            cout << "BUS FROM L1CACHE [" << id
                 << "]: BUS SEND REQ TO L2 FROM L1"
                 << " Time stamp: " << sc_time_stamp() << " Core ID "
                 << sourceId << " Address: " << trans.get_address() << endl;
#endif

            // 将请求加入队列
            BusRequest request(trans.get_address(), reqType, sourceId, &trans);
            requestQueue.push(request);
            newRequest.notify();

            return TLM_ACCEPTED;
        } else if (phase == END_RESP) {
            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = END_RESP;
#if GPU_CACHE_DEBUG == 1
            cout << "BUS FROM L1CACHE [" << id << "]: BUS SEND END RESP TO L2"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif
            l2_socket->nb_transport_fw(trans, phase, delay);
            return TLM_COMPLETED;
        } else {
            assert(false);
        }

        return TLM_ACCEPTED;
    }

    // 处理L2发来的响应
    tlm_sync_enum nb_transport_bw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {
        if (phase == BEGIN_RESP) {

            // 获取源ID扩展
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);

            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
                return TLM_COMPLETED;
            }

            int targetId = id_ext->get_id();
#if GPU_CACHE_DEBUG == 1
            cout << "BUS FROM L1CACHE [" << targetId
                 << "]: BUS BEGIN RESP FROM L2"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif

            // 我们需要直接调用 L1 缓存的 nb_transport_bw 方法
            // if (targetId >= 0 && targetId < l1Caches.size()) {
            //     l1Caches[targetId]->nb_transport_bw(trans, phase, delay);
            // } else {
            //     SC_REPORT_ERROR("Bus", "Invalid target ID");
            // }

            if (targetId >= 0 && targetId < l1Caches.size()) {
                (*l1_sockets[targetId])->nb_transport_bw(trans, phase, delay);
            } else {
                SC_REPORT_ERROR("Bus", "Invalid target ID");
            }
            // // 转发响应
            // l1_sockets[targetId]->nb_transport_bw(trans, phase, delay);
            return TLM_UPDATED;
        } else if (phase == END_REQ) {
            // 获取源ID扩展
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);

            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
                return TLM_COMPLETED;
            }
            int targetId = id_ext->get_id();
            // sc_time busDelay = SC_ZERO_TIME;
            // (*l1_sockets[targetId])->nb_transport_bw(trans, phase, busDelay);
        } else if (phase == END_RESP) {
            // 获取源ID扩展
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);


            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
                return TLM_COMPLETED;
            }
            int targetId = id_ext->get_id();
#if GPU_CACHE_DEBUG == 1
            cout << "BUS FROM L1CACHE [" << targetId
                 << "]: BUS END RESP FROM L2 HIT"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif

            sc_time busDelay = SC_ZERO_TIME;
            (*l1_sockets[targetId])->nb_transport_bw(trans, phase, busDelay);
        } else {
            assert(false);
        }

        return TLM_ACCEPTED;
    }
};


struct WriteBackExtension : public tlm::tlm_extension<WriteBackExtension> {
    WbOp op;

    WriteBackExtension(WbOp o = WbOp::MSHR) : op(o) {}

    virtual tlm_extension_base *clone() const override {
        return new WriteBackExtension(op);
    }

    virtual void copy_from(const tlm_extension_base &ext) override {
        op = static_cast<const WriteBackExtension &>(ext).op;
    }
};

// L2缓存
class L2Cache : public CacheBase {
private:
    sc_mutex requestMutex;

    // 统一的请求结构体
    struct CacheRequest {
        enum RequestType { READ_REQ, WRITE_REQ, WRITEBACK_REQ };

        RequestType type;
        uint64_t address;
        uint8_t *data;
        int dataLength;
        tlm_generic_payload *originalTrans;

        CacheRequest(RequestType t, uint64_t addr, uint8_t *d, int len,
                     tlm_generic_payload *orig = nullptr)
            : type(t),
              address(addr),
              data(d),
              dataLength(len),
              originalTrans(orig) {}
    };

    // 添加写回请求队列和事件
    struct WritebackRequest {
        uint64_t address;
        uint8_t *data;
        int lineSize;
    };
    std::queue<WritebackRequest> writebackQueue;
    sc_event newWritebackRequest;
    std::queue<CacheRequest> requestQueue;
    sc_event newRequest;

public:
    tlm_utils::simple_target_socket<L2Cache> bus_socket;
    tlm_utils::simple_initiator_socket<L2Cache> mem_socket;
    tlm_utils::peq_with_cb_and_phase<L2Cache> payloadEventQueue;
    tlm_utils::peq_with_cb_and_phase<L2Cache> payloadEventQueue_L2WB;
    tlm_utils::peq_with_cb_and_phase<L2Cache> payloadEventQueue_L2L1WB;
    sc_event end_req_event;
    sc_event mshrevent;
    sc_core::sc_time lastEndRequest = sc_core::sc_max_time();
    // Add file stream for logging
    std::ofstream logFile;

    void logCacheAccess(bool isHit, string RW, uint64_t address) {
        std::string logFileName =
            "gpu_cache/L2Cache_cid_" + std::to_string(0) + ".log";

        logFile.open(logFileName, std::ios::app);
        if (logFile.is_open()) {
            logFile << "Core " << 0 << " Rw " << RW
                    << " Time: " << sc_core::sc_time_stamp().to_string()
                    << " ns Address: 0x" << std::hex << address << std::dec
                    << " Result: " << (isHit ? "HIT" : "MISS") << std::endl;
            logFile.close();
        } else {

            assert(false);
        }
    }

    SC_HAS_PROCESS(L2Cache);

    L2Cache(sc_module_name name, int cacheSize, int lineSize, int associativity,
            int numMSHRs)
        : CacheBase(name, cacheSize, lineSize, associativity, numMSHRs),
          bus_socket("bus_socket"),
          mem_socket("mem_socket"),
          payloadEventQueue(this, &L2Cache::peqCallback),
          payloadEventQueue_L2WB(this, &L2Cache::peqCallback_L2WB),
          payloadEventQueue_L2L1WB(this, &L2Cache::peqCallback_L2L1WB) {
        std::string logFileName =
            "gpu_cache/L2Cache_cid_" + std::to_string(0) + ".log";
        logFile.open(logFileName, std::ios::app);

        if (!logFile.is_open()) {
            SC_REPORT_ERROR(
                "L2Cache", ("Failed to open log file: " + logFileName).c_str());
        }

        bus_socket.register_nb_transport_fw(this, &L2Cache::nb_transport_fw);
        mem_socket.register_nb_transport_bw(this, &L2Cache::nb_transport_bw);
        SC_THREAD(processMSHRs);
        SC_THREAD(writebackHandler);    // 注册写回处理线程
        SC_THREAD(processRequestQueue); // 注册统一请求处理线程
    }

    ~L2Cache() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {

        if (phase == tlm::END_RESP) {
            tlm_phase l2Phase = END_RESP;
            sc_time l2Delay = SC_ZERO_TIME;
            SourceIDExtension *id_ext;
            payload.get_extension(id_ext);


            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
            }
            int targetId = id_ext->get_id();
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE FROM L1CACHE [" << targetId
                 << "]: END RESP For L1CACHE MSHR"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << payload.get_address() << endl;
#endif

            bus_socket->nb_transport_bw(payload, l2Phase, l2Delay);
        } else if (phase == tlm::END_REQ) {

            tlm_phase l2Phase = END_REQ;
            sc_time l2Delay = SC_ZERO_TIME;

            bus_socket->nb_transport_bw(payload, l2Phase, l2Delay);
        } else {
            assert(false);
        }
    }
    void peqCallback_L2WB(tlm::tlm_generic_payload &payload,
                          const tlm::tlm_phase &phase) {

        if (phase == tlm::END_RESP) {
            tlm_phase l2Phase = END_RESP;
            sc_time l2Delay = SC_ZERO_TIME;

            mem_socket->nb_transport_fw(payload, l2Phase, l2Delay);
        }
    }

    void peqCallback_L2L1WB(tlm::tlm_generic_payload &payload,
                            const tlm::tlm_phase &phase) {

        if (phase == tlm::END_RESP) {
            tlm_phase l2Phase = END_RESP;
            sc_time l2Delay = SC_ZERO_TIME;

            mem_socket->nb_transport_fw(payload, l2Phase, l2Delay);
#if MSHRHIT == 2

#else
            bus_socket->nb_transport_bw(payload, l2Phase, l2Delay);
#endif
        }
    }

    // 修改writeBack方法，将写回请求加入队列
    void writeBack(uint64_t address, int setIndex, int index) {
#if GPU_CACHE_DEBUG == 1
        cout << "L2CACHE WB." << " Time stamp: " << sc_time_stamp()
             << " Address: " << address << endl;
#endif


        // 创建写回请求并加入队列
        WritebackRequest req;
        req.address = address;
        req.data = new uint8_t[lineSize]; // 分配新的内存以避免数据被覆盖
        // memcpy(req.data, &sets[setIndex].lines[index].data[0], lineSize);
        req.lineSize = lineSize;

        writebackQueue.push(req);
        newWritebackRequest.notify(); // 通知写回处理线程

        // 标记为非脏
        sets[setIndex].lines[index].dirty = false;
    }

    // 添加写回处理线程
    void writebackHandler() {
        while (true) {
            if (writebackQueue.empty()) {
#if DEBUG == 1
                SC_REPORT_INFO("Event Wait,(Waiting for for L2: ",
                               newWritebackRequest.name());
#endif

                wait(newWritebackRequest); // 等待新的写回请求
            }

            // 处理队列中的写回请求
            WritebackRequest req = writebackQueue.front();
            writebackQueue.pop();
            // 将写回请求添加到统一队列
            CacheRequest unifiedReq(CacheRequest::WRITEBACK_REQ, req.address,
                                    req.data, req.lineSize);
            requestQueue.push(unifiedReq);
            newRequest.notify(); // 通知请求处理线程
        }
    }

    // 修改processMSHRs方法，将MSHR请求转发到统一队列
    void processMSHRs() {
        while (true) {
            // 检查并处理MSHR中的请求
#if DEBUG == 1
            SC_REPORT_INFO("Event Wait,(Waiting for for: ", "L2MSHR");
#endif
            bool tmp = false;

            for (int i = 0; i < numMSHRs; i++) {
                if (mshrEntries[i].isIssue == false) {
                    tmp = true;
                    break;
                }
            }
            if (tmp == false) {
                wait(mshrevent);
            }
            for (int i = 0; i < numMSHRs; i++) {
                if (mshrEntries[i].isIssue == false) {
                    // 处理MSHR中的请求...

                    mshrEntries[i].isIssue = true;
                    tlm_generic_payload *trans =
                        mshrEntries[i].pendingTransaction;
                    uint64_t addr = trans->get_address();

                    if (mshrEntries[i].requestType == READ) {
                        // 创建读请求并加入统一队列
                        uint8_t *data = new uint8_t[trans->get_data_length()];
                        // memcpy(data, trans->get_data_ptr(),
                        // trans->get_data_length());

                        CacheRequest req(CacheRequest::READ_REQ, addr, data,
                                         trans->get_data_length(), trans);
                        requestQueue.push(req);
                        newRequest.notify();

                    } else if (mshrEntries[i].requestType == WRITE ||
                               mshrEntries[i].requestType == WRITEBACK) {
                        // 创建写请求并加入统一队列
                        uint8_t *data = new uint8_t[trans->get_data_length()];
                        // memcpy(data, trans->get_data_ptr(),
                        // trans->get_data_length());

                        CacheRequest req(CacheRequest::WRITE_REQ, addr, data,
                                         trans->get_data_length(), trans);
                        requestQueue.push(req);
                        newRequest.notify();
                    } else {
                        assert(false);
                    }
                }
            }
            // wait(CYCLE, SC_NS); // 定期检查
        }
    }


    // 添加新的统一请求处理线程
    void processRequestQueue() {
        while (true) {
            if (requestQueue.empty()) {
#if DEBUG == 1
                SC_REPORT_INFO("Event Wait,(Waiting for for L2: ",
                               newRequest.name());
#endif
                wait(newRequest); // 等待新请求
            }

            // 处理队列中的请求
            CacheRequest req = requestQueue.front();
            requestQueue.pop();

            // 创建事务
            tlm_generic_payload *newTrans = req.originalTrans;

            switch (req.type) {
            case CacheRequest::READ_REQ: {
                newTrans->set_command(TLM_READ_COMMAND);
                WriteBackExtension *op_ext = new WriteBackExtension(WbOp::MSHR);
                newTrans->set_extension(op_ext);
                break;
            }

            case CacheRequest::WRITE_REQ: {
                newTrans->set_command(TLM_WRITE_COMMAND);
                WriteBackExtension *op_ext = new WriteBackExtension(WbOp::MSHR);
                newTrans->set_extension(op_ext);
                break;
            }
            case CacheRequest::WRITEBACK_REQ: {
                // 为写回请求创建新的事务
                tlm_generic_payload *newTrans2 = &(mm.allocate(req.dataLength));
                newTrans = newTrans2;
                newTrans->acquire();
                newTrans->set_command(TLM_WRITE_COMMAND);
                // newTrans->set_address(req.address);
                // newTrans->set_data_ptr(req.data);
                // newTrans->set_data_length(req.dataLength);
                WriteBackExtension *op_ext = new WriteBackExtension(WbOp::WB);
                newTrans->set_extension(op_ext);

                break;
            }
            }

            // 检查req是否为空
            if (!req.data || !req.dataLength) {
                if (!req.address) {
                    cout << "Address is null." << req.address << endl;
                    cout << "Error: Request address is null." << endl;
                }
                if (!req.data) {
                    cout << "Error: Request data pointer is null." << endl;
                }
                if (!req.dataLength) {
                    cout << "Error: Request data length is zero." << endl;
                }
                if (!req.address || !req.data || !req.dataLength) {
                    SC_REPORT_ERROR("L2Cache", "Invalid request parameters");
                    return;
                }
                SC_REPORT_ERROR("L2Cache", "Invalid request parameters");
                return;
            }

            // 检查newTrans是否为空
            if (!newTrans) {
                SC_REPORT_ERROR("L2Cache", "Invalid transaction pointer");
                return;
            }

            // 设置事务参数
            newTrans->set_address(req.address);
            newTrans->set_data_ptr(req.data);
            newTrans->set_data_length(req.dataLength);

            // 在下一个时钟周期发送请求
            wait(SC_ZERO_TIME); // 等待当前delta周期结束

            // 发送请求
            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = BEGIN_REQ;

            sc_core::sc_time clkPeriod;
            clkPeriod = sc_core::sc_time(CYCLE, sc_core::SC_NS);

            sc_core::sc_time sendingTime = sc_core::sc_time_stamp() + delay;

            bool needsOffset =
                (sendingTime % clkPeriod) != sc_core::SC_ZERO_TIME;
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
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE SEND REQ TO MEMORY."
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << req.address << endl;
#endif

            mem_socket->nb_transport_fw(*newTrans, phase, delay);

            // 等待响应（简化处理）
#if DEBUG == 1
            SC_REPORT_INFO("Event Wait,(Waiting for for L2: ",
                           end_req_event.name());
#endif

            wait(end_req_event);

            // 如果是写回请求，释放数据内存
            if (req.type == CacheRequest::WRITEBACK_REQ) {
                delete[] req.data;
            }
        }
    }
    // 处理总线发来的请求
    tlm_sync_enum nb_transport_fw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {
        if (phase == BEGIN_REQ) {
            // 加锁确保一次只处理一个请求
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);

            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
            }
            requestMutex.lock();

            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);

            if (trans.get_command() == TLM_READ_COMMAND) {
                // 检查是否命中
                bool hit = false;
                for (auto &line : sets[setIndex].lines) {
                    if (line.valid && line.tag == tag) {
                        hit = true;
                        if (GPU_CACHE_LOG) {
                            logCacheAccess(true, "READ", addr);
                        }
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: READ HIT."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag << endl;
#endif
                        // 读取数据
                        // memcpy(trans.get_data_ptr(), &line.data[0],
                        // trans.get_data_length());
                        // delay += sc_time(10, SC_NS); // L2命中延迟

                        requestMutex.unlock();
                        phase = END_RESP;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        return TLM_UPDATED;
                    }
                }

                if (!hit) {
                    if (GPU_CACHE_LOG) {
                        logCacheAccess(false, "READ", addr);
                    }
#if GPU_CACHE_DEBUG == 1
                    cout << "L2Cache: READ MISS."
                         << " Time stamp: " << sc_time_stamp()
                         << " Address: " << addr << "Tag: " << tag << endl;
#endif
#if MSHRHIT == 1
                    bool mshr_hit = false;
                    uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                    for (int i = 0; i < mshrEntries.size(); i++) {
                        if (mshrEntries[i].isPending == true) {
                            parseAddress(mshrEntries[i].address, mshr_tag,
                                         mshr_setIndex, mshr_offset);
                            if (mshr_tag == tag && mshr_setIndex == setIndex) {

                                mshr_hit = true;
                                requestMutex.unlock();
                                phase = END_RESP;
                                sc_time bwDelay =
                                    sc_core::sc_time(CYCLE, sc_core::SC_NS);
                                payloadEventQueue.notify(trans, phase, bwDelay);
                                return TLM_UPDATED;
                            }
                        }
                    }


#endif
                    // 未命中，放入MSHR
                    int mshrIndex = findFreeMSHR();
                    // #if GPU_CACHE_DEBUG == 1

                    //                     cout << "MSHR Entries Status:" <<
                    //                     endl; for (int i = 0; i < numMSHRs;
                    //                     i++) {
                    //                         cout << "MSHR " << i << ": "
                    //                             << "Address: " <<
                    //                             mshrEntries[i].address << "
                    //                             Type: "
                    //                             <<
                    //                             (mshrEntries[i].requestType
                    //                             == READ ? "READ"
                    //                                 :
                    //                                 mshrEntries[i].requestType
                    //                                 == WRITE ? "WRITE" :
                    //                                 "WRITEBACK")
                    //                             << " Time: " <<
                    //                             mshrEntries[i].requestTime <<
                    //                             " Pending: "
                    //                             << (mshrEntries[i].isPending
                    //                             ? "Yes" : "No")
                    //                             << " Issue: " <<
                    //                             (mshrEntries[i].isIssue ?
                    //                             "Yes" : "No")
                    //                             << endl;
                    //                     }
                    //                     cout << "MSHR Index: " << mshrIndex
                    //                     << " READ COMMAND " << endl;
                    // #endif
                    if (mshrIndex >= 0) {
                        bool mshr_hit = false;
                        int i_index;
#if MSHRHIT == 2

                        uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                        for (int i = 0; i < mshrEntries.size(); i++) {
                            if (mshrEntries[i].isPending == true) {
                                parseAddress(mshrEntries[i].address, mshr_tag,
                                             mshr_setIndex, mshr_offset);
                                if (mshr_tag == tag &&
                                    mshr_setIndex == setIndex) {

                                    mshr_hit = true;
                                    i_index = i;
                                    break;
                                }
                            }
                        }
#endif
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: MSHR hit." << mshr_hit
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag
                             << "MSHR Address: " << mshrEntries[i_index].address
                             << " Tag: " << mshr_tag
                             << " Set Index: " << mshr_setIndex << endl;
#endif
                        mshrEntries[mshrIndex].address = addr;
                        mshrEntries[mshrIndex].requestType = READ;
                        mshrEntries[mshrIndex].requestTime = sc_time_stamp();
                        mshrEntries[mshrIndex].pendingTransaction = &trans;
                        mshrEntries[mshrIndex].isPending = true;
                        mshrEntries[mshrIndex].isIssue = false;
                        if (mshr_hit == true) {
                            mshrEntries[mshrIndex].isIssue = true;
                        }

                        requestMutex.unlock();
                        phase = END_REQ;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        if (mshr_hit == false) {
                            mshrevent.notify();
                        }

                        return TLM_ACCEPTED;
                    } else {
                        // MSHR已满，拒绝请求
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: read MSHR full."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag << endl;
#endif
                        requestMutex.unlock();
                        trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
                        return TLM_COMPLETED;
                    }
                }
            } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                // 写请求逻辑
                bool hit = false;
                for (auto &line : sets[setIndex].lines) {
                    if (line.valid && line.tag == tag) {
                        hit = true;
                        if (GPU_CACHE_LOG) {
                            logCacheAccess(true, "WRITE", addr);
                        }
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: WRITE HIT."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag << endl;
#endif
                        // 写入数据
                        // memcpy(&line.data[0], trans.get_data_ptr(),
                        // trans.get_data_length());
                        // delay += sc_time(10, SC_NS);

                        requestMutex.unlock();
                        phase = END_RESP;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        return TLM_UPDATED;
                    }
                }

                if (!hit) {
                    if (GPU_CACHE_LOG) {
                        logCacheAccess(false, "WRITE", addr);
                    }
#if GPU_CACHE_DEBUG == 1
                    cout << "L2Cache: WRITE MISS."
                         << " Time stamp: " << sc_time_stamp()
                         << " Address: " << addr << "Tag: " << tag << endl;
#endif
#if MSHRHIT == 1
                    bool mshr_hit = false;
                    uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                    for (int i = 0; i < mshrEntries.size(); i++) {
                        if (mshrEntries[i].isPending == true) {
                            parseAddress(mshrEntries[i].address, mshr_tag,
                                         mshr_setIndex, mshr_offset);
                            if (mshr_tag == tag && mshr_setIndex == setIndex) {

                                mshr_hit = true;
                                requestMutex.unlock();
                                phase = END_RESP;
                                sc_time bwDelay =
                                    sc_core::sc_time(CYCLE, sc_core::SC_NS);
                                payloadEventQueue.notify(trans, phase, bwDelay);
                                return TLM_UPDATED;
                            }
                        }
                    }


#endif
                    int mshrIndex = findFreeMSHR();
                    // #if GPU_CACHE_DEBUG == 1

                    //                     cout << "MSHR Entries Status:" <<
                    //                     endl; for (int i = 0; i < numMSHRs;
                    //                     i++) {
                    //                         cout << "MSHR " << i << ": "
                    //                             << "Address: " <<
                    //                             mshrEntries[i].address << "
                    //                             Type: "
                    //                             <<
                    //                             (mshrEntries[i].requestType
                    //                             == READ ? "READ"
                    //                                 :
                    //                                 mshrEntries[i].requestType
                    //                                 == WRITE ? "WRITE" :
                    //                                 "WRITEBACK")
                    //                             << " Time: " <<
                    //                             mshrEntries[i].requestTime <<
                    //                             " Pending: "
                    //                             << (mshrEntries[i].isPending
                    //                             ? "Yes" : "No")
                    //                             << " Issue: " <<
                    //                             (mshrEntries[i].isIssue ?
                    //                             "Yes" : "No")
                    //                             << endl;
                    //                     }
                    //                     cout << "MSHR Index: " << mshrIndex
                    //                     << " WRITE COMMAND " << endl;
                    // #endif
                    if (mshrIndex >= 0) {
                        bool mshr_hit = false;
                        int i_index;
#if MSHRHIT == 2

                        uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                        for (int i = 0; i < mshrEntries.size(); i++) {
                            if (mshrEntries[i].isPending == true) {
                                parseAddress(mshrEntries[i].address, mshr_tag,
                                             mshr_setIndex, mshr_offset);
                                if (mshr_tag == tag &&
                                    mshr_setIndex == setIndex) {

                                    mshr_hit = true;
                                    i_index = i;
                                    break;
                                }
                            }
                        }
#endif
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: WRITE MSHR hit." << mshr_hit
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag
                             << " setIndex: " << setIndex
                             << "MSHR Address: " << mshrEntries[i_index].address
                             << " Tag: " << mshr_tag
                             << " Set Index: " << mshr_setIndex << endl;

#endif
                        mshrEntries[mshrIndex].address = addr;
                        mshrEntries[mshrIndex].requestType = WRITE;
                        mshrEntries[mshrIndex].requestTime = sc_time_stamp();
                        mshrEntries[mshrIndex].pendingTransaction = &trans;
                        mshrEntries[mshrIndex].isPending = true;
                        mshrEntries[mshrIndex].isIssue = false;
                        if (mshr_hit == true) {
                            mshrEntries[mshrIndex].isIssue = true;
                        }

                        requestMutex.unlock();
                        phase = END_REQ;
                        sc_time bwDelay =
                            sc_core::sc_time(CYCLE, sc_core::SC_NS);
                        payloadEventQueue.notify(trans, phase, bwDelay);
                        if (mshr_hit == false) {
                            mshrevent.notify();
                        }

                        return TLM_ACCEPTED;
                    } else {
#if GPU_CACHE_DEBUG == 1
                        cout << "L2Cache: write MSHR full."
                             << " Time stamp: " << sc_time_stamp()
                             << " Address: " << addr << "Tag: " << tag << endl;
#endif
                        requestMutex.unlock();
                        trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
                        return TLM_COMPLETED;
                    }
                }
            } else {
                assert(false);
            }

            requestMutex.unlock();
        } else if (phase == END_RESP) {
            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = END_RESP;
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE END RESP." << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif

            mem_socket->nb_transport_fw(trans, phase, delay);
            return TLM_COMPLETED;
        }

        return TLM_ACCEPTED;
    }

    // 处理主存发来的响应
    tlm_sync_enum nb_transport_bw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {
        WriteBackExtension *op_ext = trans.get_extension<WriteBackExtension>();
        if (!op_ext) {
            SC_REPORT_FATAL("L2Cache", "Missing WB Type");
            return TLM_COMPLETED; // 修改为返回
        }

        WriteBackL1Extension *opL1_ext =
            trans.get_extension<WriteBackL1Extension>();
        if (!opL1_ext) {
            opL1_ext = new WriteBackL1Extension(WbL1Op::MSHR_L1);
        }

        if (phase == END_REQ) {
            end_req_event.notify();
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE END REQ." << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif
            lastEndRequest = sc_core::sc_time_stamp();

            // sc_time busDelay = SC_ZERO_TIME;
            // bus_socket->nb_transport_bw(trans, phase, busDelay);
        } else if (phase == BEGIN_RESP && op_ext->op == WbOp::MSHR &&
                   opL1_ext->op == WbL1Op::MSHR_L1) {
            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);


            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
            }
            int targetId = id_ext->get_id();

#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE FROM L1CACHE [" << targetId
                 << "]: BESP RESP For L1CACHE MSHR"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;

            // 打印所有MSHR条目的信息
            cout << "MSHR Entries Status:" << endl;
            for (int i = 0; i < numMSHRs; i++) {
                cout << "MSHR " << i << ": "
                     << "Address: " << mshrEntries[i].address << " Type: "
                     << (mshrEntries[i].requestType == READ    ? "READ"
                         : mshrEntries[i].requestType == WRITE ? "WRITE"
                                                               : "WRITEBACK")
                     << " Time: " << mshrEntries[i].requestTime << " Pending: "
                     << (mshrEntries[i].isPending ? "Yes" : "No")
                     << " Issue: " << (mshrEntries[i].isIssue ? "Yes" : "No")
                     << endl;
            }
            cout << "address: " << addr << " tag: " << tag
                 << " setIndex: " << setIndex << " offset: " << offset << endl;
#endif
            // 查找对应的MSHR
            int mshrIndex = findMSHRByAddress(addr);
            assert(mshrIndex >= 0 && "MSHR not found");
            if (mshrIndex >= 0) {
#if MSHRHIT == 2
                bool mshr_hit = false;
                uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                int tmp_cout = 0;
                for (int mshr_i = 0; mshr_i < mshrEntries.size(); mshr_i++) {
                    if (mshrEntries[mshr_i].isPending == true) {
                        parseAddress(mshrEntries[mshr_i].address, mshr_tag,
                                     mshr_setIndex, mshr_offset);
                        if (mshr_tag == tag && mshr_setIndex == setIndex) {

                            mshr_hit = true;
                            tmp_cout++;
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE  L2 mshr l1 mshr tmp_cout: "
                                 << tmp_cout << endl;
#endif
                            auto mshr_trans =
                                mshrEntries[mshr_i].pendingTransaction;
                            if (mshr_trans->get_command() == TLM_READ_COMMAND) {
                                // 处理读响应
                                // 为缓存行分配空间
                                int replaceIndex = -1;
                                for (int i = 0; i < associativity; i++) {
                                    if (!sets[mshr_setIndex].lines[i].valid) {
                                        replaceIndex = i;
                                        break;
                                    }
                                }

                                if (replaceIndex < 0) {
                                    replaceIndex = 0;
                                    if (sets[mshr_setIndex]
                                            .lines[replaceIndex]
                                            .state == MODIFIED) {
                                        // 计算写回地址
                                        const int log2LineSize =
                                            static_cast<int>(log2(lineSize));
                                        const int log2NumSets =
                                            static_cast<int>(log2(numSets));

                                        uint64_t writebackAddr =
                                            (sets[mshr_setIndex]
                                                 .lines[replaceIndex]
                                                 .tag
                                             << (log2LineSize + log2NumSets)) |
                                            (mshr_setIndex << log2LineSize);

                                        // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                                        cout << "L2CACHE read WRITEBACK: "
                                             << writebackAddr << endl;
#endif
                                        writeBack(writebackAddr, mshr_setIndex,
                                                  replaceIndex);
                                    }
                                }

                                // 更新缓存行
                                sets[mshr_setIndex].lines[replaceIndex].tag =
                                    mshr_tag;
                                sets[mshr_setIndex].lines[replaceIndex].valid =
                                    true;
                                sets[mshr_setIndex].lines[replaceIndex].state =
                                    SHARED;
                                // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                                // trans.get_data_ptr(), lineSize);

                                // 更新原始事务数据
                                tlm_generic_payload *origTrans =
                                    mshr_trans; // mshrEntries[mshrIndex].pendingTransaction;
                                // memcpy(origTrans->get_data_ptr(),
                                // trans.get_data_ptr(),
                                // origTrans->get_data_length());

                                // 标记MSHR为空闲
                                mshrEntries[mshr_i].isPending = false;

                                // 回复总线
                                tlm_phase busPhase = END_RESP;
                                if (origTrans == &trans) {
                                    busPhase = BEGIN_RESP;
                                }
                                sc_time busDelay = SC_ZERO_TIME;
                                bus_socket->nb_transport_bw(*origTrans,
                                                            busPhase, busDelay);
                            } else if (mshr_trans->get_command() ==
                                       TLM_WRITE_COMMAND) {
                                // 处理写响应
                                tlm_generic_payload *origTrans =
                                    mshr_trans; // mshrEntries[mshrIndex].pendingTransaction;

                                int replaceIndex = -1;
                                for (int i = 0; i < associativity; i++) {
                                    if (!sets[mshr_setIndex].lines[i].valid) {
                                        replaceIndex = i;
                                        break;
                                    }
                                }

                                if (replaceIndex < 0) {
                                    replaceIndex = 0;
                                    if (sets[mshr_setIndex]
                                            .lines[replaceIndex]
                                            .state == MODIFIED) {
                                        // 计算写回地址
                                        const int log2LineSize =
                                            static_cast<int>(log2(lineSize));
                                        const int log2NumSets =
                                            static_cast<int>(log2(numSets));

                                        uint64_t writebackAddr =
                                            (sets[mshr_setIndex]
                                                 .lines[replaceIndex]
                                                 .tag
                                             << (log2LineSize + log2NumSets)) |
                                            (mshr_setIndex << log2LineSize);

                                        // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                                        cout << "L2CACHE write WRITEBACK: "
                                             << writebackAddr << endl;
#endif
                                        writeBack(writebackAddr, mshr_setIndex,
                                                  replaceIndex);
                                    }
                                }

                                sets[mshr_setIndex].lines[replaceIndex].tag =
                                    mshr_tag;
                                sets[mshr_setIndex].lines[replaceIndex].valid =
                                    true;
                                sets[mshr_setIndex].lines[replaceIndex].state =
                                    MODIFIED;
                                // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                                // origTrans->get_data_ptr(),
                                // origTrans->get_data_length());

                                mshrEntries[mshr_i].isPending = false;

                                tlm_phase busPhase = END_RESP;
                                if (origTrans == &trans) {
                                    busPhase = BEGIN_RESP;
                                }
                                sc_time busDelay = SC_ZERO_TIME;
                                bus_socket->nb_transport_bw(*origTrans,
                                                            busPhase, busDelay);
                            } else {
                                assert(false);
                            }
                        }
                    }
                }


#else
                if (trans.get_command() == TLM_READ_COMMAND) {
                    // 处理读响应
                    // 为缓存行分配空间
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE read WRITEBACK: " << writebackAddr
                                 << endl;
#endif
                            writeBack(writebackAddr, setIndex, replaceIndex);
                        }
                    }

                    // 更新缓存行
                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = SHARED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // trans.get_data_ptr(), lineSize);

                    // 更新原始事务数据
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(origTrans->get_data_ptr(), trans.get_data_ptr(),
                    // origTrans->get_data_length());

                    // 标记MSHR为空闲
                    mshrEntries[mshrIndex].isPending = false;

                    // 回复总线
                    tlm_phase busPhase = BEGIN_RESP;
                    sc_time busDelay = SC_ZERO_TIME;
                    bus_socket->nb_transport_bw(*origTrans, busPhase, busDelay);
                } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                    // 处理写响应
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;

                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE write WRITEBACK: " << writebackAddr
                                 << endl;
#endif
                            writeBack(writebackAddr, setIndex, replaceIndex);
                        }
                    }

                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = MODIFIED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // origTrans->get_data_ptr(), origTrans->get_data_length());

                    mshrEntries[mshrIndex].isPending = false;

                    tlm_phase busPhase = BEGIN_RESP;
                    sc_time busDelay = SC_ZERO_TIME;
                    bus_socket->nb_transport_bw(*origTrans, busPhase, busDelay);
                }
#endif
            }

            return TLM_ACCEPTED;
        } else if (phase == BEGIN_RESP && op_ext->op == WbOp::WB &&
                   opL1_ext->op != WbL1Op::WB_L1) {

            // DAHU 释放L2 WB 数据
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE L2WB." << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif
            phase = END_RESP;
            sc_time bwDelay = sc_core::sc_time(CYCLE, sc_core::SC_NS);
            payloadEventQueue_L2WB.notify(trans, phase, bwDelay);


        } else if (phase == BEGIN_RESP && op_ext->op != WbOp::WB &&
                   opL1_ext->op == WbL1Op::WB_L1) {

            uint64_t addr = trans.get_address();
            uint64_t tag, setIndex, offset;
            parseAddress(addr, tag, setIndex, offset);
            SourceIDExtension *id_ext;
            trans.get_extension(id_ext);


            if (!id_ext) {
                SC_REPORT_ERROR("Bus", "Missing source ID extension");
            }
            int targetId = id_ext->get_id();

#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE FROM L1CACHE [" << targetId
                 << "]: BESP RESP For L1CACHE MSHR"
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;

            // 打印所有MSHR条目的信息
            cout << "MSHR WB_L1 Entries Status:" << endl;
            for (int i = 0; i < numMSHRs; i++) {
                cout << "MSHR " << i << ": "
                     << "Address: " << mshrEntries[i].address << " Type: "
                     << (mshrEntries[i].requestType == READ    ? "READ"
                         : mshrEntries[i].requestType == WRITE ? "WRITE"
                                                               : "WRITEBACK")
                     << " Time: " << mshrEntries[i].requestTime << " Pending: "
                     << (mshrEntries[i].isPending ? "Yes" : "No")
                     << " Issue: " << (mshrEntries[i].isIssue ? "Yes" : "No")
                     << endl;
            }
            cout << "address: " << addr << " tag: " << tag
                 << " setIndex: " << setIndex << " offset: " << offset << endl;
#endif
            // 查找对应的MSHR
            int mshrIndex = findMSHRByAddress(addr);
            assert(mshrIndex >= 0 && "MSHR not found");
            if (mshrIndex >= 0) {
#if MSHRHIT == 2
                bool mshr_hit = false;
                uint64_t mshr_tag, mshr_setIndex, mshr_offset;
                int tmp_cout = 0;
                for (int mshr_i = 0; mshr_i < mshrEntries.size(); mshr_i++) {
                    if (mshrEntries[mshr_i].isPending == true) {
                        parseAddress(mshrEntries[mshr_i].address, mshr_tag,
                                     mshr_setIndex, mshr_offset);
                        if (mshr_tag == tag && mshr_setIndex == setIndex) {

                            mshr_hit = true;
                            tmp_cout++;
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE  L2 mshr l1 wb tmp_cout: "
                                 << tmp_cout << endl;
#endif
                            auto mshr_trans =
                                mshrEntries[mshr_i].pendingTransaction;
                            if (mshr_trans->get_command() == TLM_READ_COMMAND) {
                                // 处理读响应
                                // 为缓存行分配空间
                                int replaceIndex = -1;
                                for (int i = 0; i < associativity; i++) {
                                    if (!sets[mshr_setIndex].lines[i].valid) {
                                        replaceIndex = i;
                                        break;
                                    }
                                }

                                if (replaceIndex < 0) {
                                    replaceIndex = 0;
                                    if (sets[mshr_setIndex]
                                            .lines[replaceIndex]
                                            .state == MODIFIED) {
                                        // 计算写回地址
                                        const int log2LineSize =
                                            static_cast<int>(log2(lineSize));
                                        const int log2NumSets =
                                            static_cast<int>(log2(numSets));

                                        uint64_t writebackAddr =
                                            (sets[mshr_setIndex]
                                                 .lines[replaceIndex]
                                                 .tag
                                             << (log2LineSize + log2NumSets)) |
                                            (mshr_setIndex << log2LineSize);

                                        // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                                        cout << "L2CACHE read WRITEBACK: "
                                             << writebackAddr << endl;
#endif
                                        writeBack(writebackAddr, mshr_setIndex,
                                                  replaceIndex);
                                    }
                                }

                                // 更新缓存行
                                sets[mshr_setIndex].lines[replaceIndex].tag =
                                    mshr_tag;
                                sets[mshr_setIndex].lines[replaceIndex].valid =
                                    true;
                                sets[mshr_setIndex].lines[replaceIndex].state =
                                    SHARED;
                                // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                                // trans.get_data_ptr(), lineSize);

                                // 更新原始事务数据
                                tlm_generic_payload *origTrans =
                                    mshr_trans; // mshrEntries[mshrIndex].pendingTransaction;
                                // memcpy(origTrans->get_data_ptr(),
                                // trans.get_data_ptr(),
                                // origTrans->get_data_length());

                                // 标记MSHR为空闲
                                mshrEntries[mshr_i].isPending = false;

                                // 回复总线
                                tlm_phase busPhase = END_RESP;
                                sc_time busDelay = SC_ZERO_TIME;
                                bus_socket->nb_transport_bw(*origTrans,
                                                            busPhase, busDelay);
                            } else if (mshr_trans->get_command() ==
                                       TLM_WRITE_COMMAND) {
                                // 处理写响应
                                tlm_generic_payload *origTrans =
                                    mshr_trans; // mshrEntries[mshrIndex].pendingTransaction;

                                int replaceIndex = -1;
                                for (int i = 0; i < associativity; i++) {
                                    if (!sets[mshr_setIndex].lines[i].valid) {
                                        replaceIndex = i;
                                        break;
                                    }
                                }

                                if (replaceIndex < 0) {
                                    replaceIndex = 0;
                                    if (sets[mshr_setIndex]
                                            .lines[replaceIndex]
                                            .state == MODIFIED) {
                                        // 计算写回地址
                                        const int log2LineSize =
                                            static_cast<int>(log2(lineSize));
                                        const int log2NumSets =
                                            static_cast<int>(log2(numSets));

                                        uint64_t writebackAddr =
                                            (sets[mshr_setIndex]
                                                 .lines[replaceIndex]
                                                 .tag
                                             << (log2LineSize + log2NumSets)) |
                                            (mshr_setIndex << log2LineSize);

                                        // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                                        cout << "L2CACHE write WRITEBACK: "
                                             << writebackAddr << endl;
#endif
                                        writeBack(writebackAddr, mshr_setIndex,
                                                  replaceIndex);
                                    }
                                }

                                sets[mshr_setIndex].lines[replaceIndex].tag =
                                    mshr_tag;
                                sets[mshr_setIndex].lines[replaceIndex].valid =
                                    true;
                                sets[mshr_setIndex].lines[replaceIndex].state =
                                    MODIFIED;
                                // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                                // origTrans->get_data_ptr(),
                                // origTrans->get_data_length());

                                mshrEntries[mshr_i].isPending = false;

                                tlm_phase busPhase = END_RESP;
                                sc_time busDelay = SC_ZERO_TIME;
                                bus_socket->nb_transport_bw(*origTrans,
                                                            busPhase, busDelay);
                            } else {
                                assert(false);
                            }
                        }
                    }
                }


#else
                if (trans.get_command() == TLM_READ_COMMAND) {
                    // 处理读响应
                    // 为缓存行分配空间
                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE read WRITEBACK: " << writebackAddr
                                 << endl;
#endif
                            writeBack(writebackAddr, setIndex, replaceIndex);
                        }
                    }

                    // 更新缓存行
                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = SHARED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // trans.get_data_ptr(), lineSize);

                    // 更新原始事务数据
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;
                    // memcpy(origTrans->get_data_ptr(), trans.get_data_ptr(),
                    // origTrans->get_data_length());

                    // 标记MSHR为空闲
                    mshrEntries[mshrIndex].isPending = false;

                } else if (trans.get_command() == TLM_WRITE_COMMAND) {
                    // 处理写响应
                    tlm_generic_payload *origTrans =
                        &trans; // mshrEntries[mshrIndex].pendingTransaction;

                    int replaceIndex = -1;
                    for (int i = 0; i < associativity; i++) {
                        if (!sets[setIndex].lines[i].valid) {
                            replaceIndex = i;
                            break;
                        }
                    }

                    if (replaceIndex < 0) {
                        replaceIndex = 0;
                        if (sets[setIndex].lines[replaceIndex].state ==
                            MODIFIED) {
                            // 计算写回地址
                            const int log2LineSize =
                                static_cast<int>(log2(lineSize));
                            const int log2NumSets =
                                static_cast<int>(log2(numSets));

                            uint64_t writebackAddr =
                                (sets[setIndex].lines[replaceIndex].tag
                                 << (log2LineSize + log2NumSets)) |
                                (setIndex << log2LineSize);

                            // 使用writeBack方法
#if GPU_CACHE_DEBUG == 1
                            cout << "L2CACHE write WRITEBACK: " << writebackAddr
                                 << endl;
#endif
                            writeBack(writebackAddr, setIndex, replaceIndex);
                        }
                    }

                    sets[setIndex].lines[replaceIndex].tag = tag;
                    sets[setIndex].lines[replaceIndex].valid = true;
                    sets[setIndex].lines[replaceIndex].state = MODIFIED;
                    // memcpy(&sets[setIndex].lines[replaceIndex].data[0],
                    // origTrans->get_data_ptr(), origTrans->get_data_length());

                    mshrEntries[mshrIndex].isPending = false;

                } else {
                    assert(false);
                }
#endif
            }

            // DAHU 释放L2 WB 数据
#if GPU_CACHE_DEBUG == 1
            cout << "L2CACHE RESP TO L1 WB."
                 << " Time stamp: " << sc_time_stamp()
                 << " Address: " << trans.get_address() << endl;
#endif

            phase = END_RESP;
            sc_time bwDelay = sc_core::sc_time(CYCLE, sc_core::SC_NS);
            payloadEventQueue_L2L1WB.notify(trans, phase, bwDelay);
        } else {
#if GPU_CACHE_DEBUG == 1
            cout << "ERROR!!!! " << endl;
#endif
        }

        return TLM_ACCEPTED;
    }
};

// 主存
class MainMemory : public sc_module {
public:
    tlm_utils::simple_target_socket<MainMemory> l2_socket;
    tlm_utils::peq_with_cb_and_phase<MainMemory> payloadEventQueue;
    tlm_utils::peq_with_cb_and_phase<MainMemory> payloadEventQueue_begin_resp;

    SC_HAS_PROCESS(MainMemory);

    MainMemory(sc_module_name name)
        : sc_module(name),
          l2_socket("l2_socket"),
          payloadEventQueue(this, &MainMemory::peqCallback),
          payloadEventQueue_begin_resp(this,
                                       &MainMemory::peqCallback_begin_resp) {
        l2_socket.register_nb_transport_fw(this, &MainMemory::nb_transport_fw);
    }


    void peqCallback_begin_resp(tlm::tlm_generic_payload &payload,
                                const tlm::tlm_phase &phase) {

        if (phase == tlm::BEGIN_RESP) {
            tlm_phase l2Phase = BEGIN_RESP;
            sc_time l2Delay = SC_ZERO_TIME;

            l2_socket->nb_transport_bw(payload, l2Phase, l2Delay);
        }
    }

    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {

        if (phase == tlm::END_REQ) {
            tlm_phase l2Phase = END_REQ;
            sc_time l2Delay = SC_ZERO_TIME;
            // SourceIDExtension* id_ext;
            // payload.get_extension(id_ext);

            // if (!id_ext) {
            //     SC_REPORT_ERROR("Bus", "Missing source ID extension");

            // }
            cout << "start main memort !!!!!!!!!" << endl;
            l2_socket->nb_transport_bw(payload, l2Phase, l2Delay);
            tlm_phase l2Phase2 = BEGIN_RESP;
            sc_time bwDelay = sc_core::sc_time(CYCLE, sc_core::SC_NS);
            payloadEventQueue_begin_resp.notify(payload, l2Phase2, bwDelay);
        }
    }
    // 处理L2发来的请求
    tlm_sync_enum nb_transport_fw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {
        if (phase == BEGIN_REQ) {
            // 模拟内存访问
            delay += sc_time(100, SC_NS); // 内存访问延迟

            // 对于读请求，生成数据
            if (trans.get_command() == TLM_READ_COMMAND) {
                uint8_t *dataPtr = trans.get_data_ptr();
                unsigned int length = trans.get_data_length();

                // 简单数据生成
                for (unsigned int i = 0; i < length; i++) {
                    dataPtr[i] = (uint8_t)(trans.get_address() + i);
                }
            }
            // SourceIDExtension* id_ext;
            // trans.get_extension(id_ext);

            // if (!id_ext) {
            //     SC_REPORT_ERROR("Bus", "Missing source ID extension");

            // }

            phase = END_REQ;
            sc_time bwDelay = sc_core::sc_time(CYCLE, sc_core::SC_NS);
            payloadEventQueue.notify(trans, phase, bwDelay);

            return TLM_UPDATED;
        } else if (phase == END_RESP) {
            return TLM_COMPLETED;
        } else {
            assert(false);
        }

        return TLM_ACCEPTED;
    }
};

// 处理器模块（简化）
class Processor : public sc_module {
private:
    uint64_t baseAddr;

public:
    tlm_utils::simple_initiator_socket<Processor> cache_socket;
    tlm_utils::peq_with_cb_and_phase<Processor> payloadEventQueue;

    SC_HAS_PROCESS(Processor);

    Processor(sc_module_name name, uint64_t baseAddr = 0)
        : sc_module(name),
          baseAddr(baseAddr),
          cache_socket("cache_socket"),
          payloadEventQueue(this, &Processor::peqCallback) {

        // SC_THREAD(generateTraffic);
        cache_socket.register_nb_transport_bw(this,
                                              &Processor::nb_transport_bw);
    }

    void peqCallback(tlm::tlm_generic_payload &payload,
                     const tlm::tlm_phase &phase) {

        if (phase == tlm::BEGIN_RESP) {

            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = END_RESP;
            cache_socket->nb_transport_fw(payload, phase, delay);
        }
    }


    tlm_sync_enum nb_transport_bw(tlm_generic_payload &trans, tlm_phase &phase,
                                  sc_time &delay) {

        if (phase == BEGIN_RESP) {

            sc_time bwDelay = sc_core::sc_time(CYCLE, sc_core::SC_NS);
            payloadEventQueue.notify(trans, phase, bwDelay);
        }
        return TLM_ACCEPTED;
    }

    void generateTraffic() {
        // 等待一段时间让系统初始化


        wait(10, SC_NS);
#if DEBUG == 1
        SC_REPORT_INFO("CPU  ", "Waiting for 10 ns");
#endif

        // 生成几个示例内存访问
        for (int i = 0; i < 10; i++) {
            // 随机决定读或写
            bool isWrite = (rand() % 2 == 0);
            // 随机地址
            uint64_t addr = baseAddr + (rand() % 1024) * 4;

            // 创建事务
            tlm_generic_payload *trans = new tlm_generic_payload();
            uint8_t data[4];

            if (isWrite) {
                // 写请求
                for (int j = 0; j < 4; j++) {
                    data[j] = rand() % 256;
                }

                trans->set_command(TLM_WRITE_COMMAND);
                trans->set_address(addr);
                trans->set_data_ptr(data);
                trans->set_data_length(4);

                cout << name() << ": Writing to address " << hex << addr << dec
                     << endl;
            } else {
                // 读请求
                trans->set_command(TLM_READ_COMMAND);
                trans->set_address(addr);
                trans->set_data_ptr(data);
                trans->set_data_length(4);

                cout << name() << ": Reading from address " << hex << addr
                     << dec << endl;
            }

            // 发送请求
            sc_time delay = SC_ZERO_TIME;
            tlm_phase phase = BEGIN_REQ;
            cache_socket->nb_transport_fw(*trans, phase, delay);

            // 等待随机时间后继续
            wait(CYCLE, SC_NS);
        }
    }
};

// 顶层模块
class CacheSystem : public sc_module {
public:
    vector<Processor *> processors;
    vector<L1Cache *> l1Caches;
    L2Cache *l2Cache;
    MainMemory *mainMemory;
    Bus *bus;

    CacheSystem(sc_module_name name, int numProcessors) : sc_module(name) {

        // 创建组件
        for (int i = 0; i < numProcessors; i++) {
            string procName = "processor_" + to_string(i);
            processors.push_back(new Processor(procName.c_str(), i * 1000));

            string l1Name = "l1_cache_" + to_string(i);
            l1Caches.push_back(new L1Cache(l1Name.c_str(), i, 8192, 64, 4, 8));
        }

        l2Cache = new L2Cache("l2_cache", 65536, 64, 8, 16);
        mainMemory = new MainMemory("main_memory");
        bus = new Bus("bus", numProcessors);

        // 连接组件
        for (int i = 0; i < numProcessors; i++) {
            processors[i]->cache_socket.bind(l1Caches[i]->cpu_socket);
            l1Caches[i]->bus_socket.bind(*bus->l1_sockets[i]);
            bus->addL1Cache(l1Caches[i]);
        }

        bus->l2_socket.bind(l2Cache->bus_socket);
        l2Cache->mem_socket.bind(mainMemory->l2_socket);
    }
};


// void processMSHRs() {
//     while (true) {
//         for (int i = 0; i < numMSHRs; i++) {
//             if (mshrEntries[i].isPending) {
//                 wait(10, SC_NS); // 模拟处理时间

//                 tlm_generic_payload* trans =
//                 mshrEntries[i].pendingTransaction; uint64_t addr =
//                 trans->get_address();

//                 if (mshrEntries[i].requestType == READ) {
//                     // 对主存发起读请求
//                     tlm_generic_payload* newTrans = new
//                     tlm_generic_payload();
//                     newTrans->set_command(TLM_READ_COMMAND);
//                     newTrans->set_address(addr);
//                     newTrans->set_data_ptr(trans->get_data_ptr());
//                     newTrans->set_data_length(trans->get_data_length());

//                     sc_time delay = SC_ZERO_TIME;
//                     tlm_phase phase = BEGIN_REQ;
//                     mem_socket->nb_transport_fw(*newTrans, phase, delay);
//                 } else if (mshrEntries[i].requestType == WRITE ||
//                 mshrEntries[i].requestType == WRITEBACK) {
//                     // 对主存发起写请求
//                     tlm_generic_payload* newTrans = new
//                     tlm_generic_payload();
//                     newTrans->set_command(TLM_WRITE_COMMAND);
//                     newTrans->set_address(addr);
//                     newTrans->set_data_ptr(trans->get_data_ptr());
//                     newTrans->set_data_length(trans->get_data_length());

//                     sc_time delay = SC_ZERO_TIME;
//                     tlm_phase phase = BEGIN_REQ;
//                     mem_socket->nb_transport_fw(*newTrans, phase, delay);
//                 }
//             }
//         }
//         wait(CYCLE, SC_NS);
//     }
// }