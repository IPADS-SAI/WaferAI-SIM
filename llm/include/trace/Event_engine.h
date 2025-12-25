#pragma once
#include "Trace_event.h"
#include "Trace_event_queue.h"
#include "systemc.h"
#include <map>
#include <vector>

class Event_engine : public sc_module {
public:
    SC_HAS_PROCESS(Event_engine);
    Event_engine(const sc_module_name &name, int trace_window);

    ~Event_engine();
    void engine_run();
    void dump_periodically();
    // void add_event(string _module_name, string _thread_name, string _type,
    // Trace_event_util _util, sc_time relative_time = sc_time(0, SC_NS));
    void add_event(string _module_name, string _thread_name, string _type,
                   Trace_event_util _util,
                   sc_time relative_time = sc_time(0, SC_NS),
                   unsigned flow_id = 0, string bp = "");
    void dump_traced_file(const string &filepath, bool is_final_dump);

private:
    void writeJsonHeader();
    void writeEvents(bool final);
    void writeJsonTail();
    bool is_first_dump;

public:
    vector<Trace_event> traced_event_list;
    Trace_event_queue Trace_event_queue_clock_engine;
    sc_event_queue sync_events;
    sc_event_queue dump_event;
    std::ofstream json_stream;
    map<string, unsigned> module_idx; // 模块名到 PID 的映射
    map<pair<string, string>, unsigned>
        thread_idx; // 模块名和线程名到 TID 的映射

    unsigned pid_count = 1;                       // PID 计数器
    map<string, unsigned> thread_count_in_module; // 模块中线程计数器
    int trace_window;                             // 追踪窗口大小
};
