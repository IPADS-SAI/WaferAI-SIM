#include "trace/Trace_event.h"

// Trace_event::Trace_event(string _module_name, string _thread_name, string
// _type, Trace_event_util _util, sc_time relative_time)
// {
// 	module_name = _module_name;
// 	thread_name = _thread_name;
// 	type = _type;
// 	util = _util;
// 	time = relative_time;
// }


void Trace_event::record_time(sc_time sim_time) { time = sim_time + time; }


Trace_event::~Trace_event() {

    /*if (clock_event != NULL)
    {

            delete clock_event;

    }*/
}
