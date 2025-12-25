#pragma once
#include "systemc.h"
#include <iostream>


using namespace std;
class Trace_event_util {
public:
    Trace_event_util() {
        m_bar_name = "";
        m_color = "None";
        m_value = 0;
    };
    Trace_event_util(float value, string color = "None")
        : m_value(value), m_color(color) {};
    Trace_event_util(string nm, string color = "None")
        : m_bar_name(nm), m_color(color) {};
    Trace_event_util(string nm, float value, string color = "None")
        : m_bar_name(nm), m_value(value), m_color(color) {}

public:
    string m_bar_name;
    string m_color;
    float m_value;
};


class Trace_event {
public:
    // Trace_event(string _module_name, string _thread_name, string _type,
    // Trace_event_util _util, sc_time relative_time = sc_time(0, SC_NS));
    Trace_event(const string &_module_name, const string &_thread_name,
                const string &_type, const Trace_event_util &_util,
                sc_time _time, unsigned _flow_id = 0, const string &_bp = "")
        : module_name(_module_name),
          thread_name(_thread_name),
          type(_type),
          util(_util),
          time(_time),
          flow_id(_flow_id),
          bp(_bp) {}
    ~Trace_event();

    void record_time(sc_time sim_time);

    string module_name;
    string thread_name;
    string type;
    Trace_event_util util;
    sc_time time;
    unsigned flow_id; // 用于 flow event
    string bp;        // 用于 flow end 的绑定点
};
