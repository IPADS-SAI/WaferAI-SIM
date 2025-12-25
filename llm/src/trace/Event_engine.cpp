#include "trace/Event_engine.h"

Event_engine::Event_engine(const sc_module_name &name, int trace_window)
    : sc_module(name), trace_window(trace_window) {
    SC_THREAD(engine_run);
    sensitive << sync_events;
    is_first_dump = true;
    SC_METHOD(dump_periodically);
    sensitive << dump_event; // 定期执行dump
    dont_initialize();
}

void Event_engine::dump_periodically() {
    dump_traced_file("events.json", false); // 写入文件
}

void Event_engine::engine_run() {
    while (1) {
        wait();
        if (Trace_event_queue_clock_engine.trace_event_queue.size() != 0) {
            Trace_event e_temp =
                Trace_event_queue_clock_engine.trace_event_queue.front();
            e_temp.record_time(sc_time_stamp());
            this->traced_event_list.push_back(e_temp);
            Trace_event_queue_clock_engine.trace_event_queue.pop();
        }
        if (traced_event_list.size() > trace_window) {
            dump_event.notify(SC_ZERO_TIME); // 触发dump_event
        }
    }
}

// void Event_engine::add_event(string _module_name, string _thread_name, string
// _type, Trace_event_util _util, sc_time relative_time)
// {
// 	Trace_event e_temp(_module_name, _thread_name, _type, _util,
// relative_time);
// 	Trace_event_queue_clock_engine.trace_event_queue.push(e_temp);
// 	sync_events.notify(SC_ZERO_TIME);
// }
void Event_engine::add_event(string _module_name, string _thread_name,
                             string _type, Trace_event_util _util,
                             sc_time relative_time, unsigned flow_id,
                             string bp) {
    Trace_event e_temp(_module_name, _thread_name, _type, _util, relative_time,
                       flow_id, bp);
    Trace_event_queue_clock_engine.trace_event_queue.push(e_temp);
    sync_events.notify(SC_ZERO_TIME);
}


void Event_engine::dump_traced_file(const string &filepath,
                                    bool is_final_dump) {

    if (is_first_dump) {
        json_stream.open(filepath);
        writeJsonHeader(); // 第一次dump
        is_first_dump = false;
    } else {
        json_stream.open(filepath,
                         std::ios::out | std::ios::app); // 打开文件以追加模式
    }
    writeEvents(is_final_dump); // 中间dump和最后dump都需要
    if (is_final_dump) {
        writeJsonTail(); // 最后一次dump
    }
    json_stream.close();

    // 清空已写入的 traced_event_list
    traced_event_list.clear();
}
// void Event_engine::dump_traced_file(const string& filepath)
// {
// 	json_stream.open(filepath);
// 	writeJsonHeader();
// 	writeEvents();
// 	writeJsonTail();
// 	json_stream.close();
// }


void Event_engine::writeJsonHeader() {
    json_stream << "{\n\"otherData\": {}, \n\"traceEvents\": [";
    json_stream.flush();
}


void Event_engine::writeJsonTail() {
    json_stream << "]\n}";
    json_stream.flush();
}


void Event_engine::writeEvents(bool final) {
    // 更新模块信息
    for (auto iter = traced_event_list.begin(); iter != traced_event_list.end();
         iter++) {
        string m_name = iter->module_name;
        if (module_idx.find(m_name) == module_idx.end()) {
            module_idx[m_name] = pid_count++;
            thread_count_in_module[m_name] = 0; // 初始化模块内的线程计数器
            // 写入模块信息
            json_stream
                << "{\"name\": \"process_name\", \"ph\": \"M\", \"pid\": ";
            json_stream << module_idx[m_name];
            json_stream << ", \"args\": {\"name\": \"";
            json_stream << m_name;
            json_stream << "\"}},\n";
        }
    }

    // 更新线程信息
    for (auto iter = traced_event_list.begin(); iter != traced_event_list.end();
         iter++) {
        string m_name = iter->module_name;
        string t_name = iter->thread_name;
        pair<string, string> m_t_name(m_name, t_name);

        if (thread_idx.find(m_t_name) == thread_idx.end()) {
            thread_idx[m_t_name] = thread_count_in_module[m_name]++;
            // 写入线程信息
            json_stream
                << "{\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": ";
            json_stream << module_idx[m_name];
            json_stream << ", \"tid\": ";
            json_stream << thread_idx[m_t_name];
            json_stream << ", \"args\": {\"name\": \"";
            json_stream << t_name;
            json_stream << "\"}},\n";
        }
    }

    // 写入事件数据
    for (auto iter = traced_event_list.begin(); iter != traced_event_list.end();
         iter++) {
        string m_name = iter->module_name;
        string t_name = iter->thread_name;
        string event_type = iter->type;
        double ts = iter->time.to_seconds() * 1e6;

        json_stream << "{\"name\": \"";
        json_stream << ((iter->util.m_bar_name != "") ? iter->util.m_bar_name
                                                      : t_name);
        json_stream << "\", \"cat\": \"";
        json_stream << m_name;
        if (iter->util.m_color != "None") {
            json_stream << "\", \"cname\": \"";
            json_stream << iter->util.m_color;
        }
        json_stream << "\", \"ph\": \"";
        json_stream << event_type;
        json_stream << "\", \"ts\": ";
        json_stream << ts;
        json_stream << ", \"pid\": ";
        json_stream << module_idx[m_name];
        json_stream << ", \"tid\": ";
        json_stream << thread_idx[make_pair(m_name, t_name)];

        // 针对 flow event 的特殊处理
        if (event_type == "s" || event_type == "f") {
            json_stream << ", \"id\": ";
            json_stream << iter->flow_id;
            if (event_type == "f" && !iter->bp.empty()) {
                json_stream << ", \"bp\": \"";
                json_stream << iter->bp;
                json_stream << "\"";
            }
        }

        // 写入通用 args 字段
        json_stream << ", \"args\": {";
        if (event_type == "C") {
            string value_name = (!iter->util.m_bar_name.empty())
                                    ? iter->util.m_bar_name
                                    : t_name;
            json_stream << "\"" << value_name << "\": ";
            json_stream << iter->util.m_value;
        } else if (!iter->util.m_bar_name.empty()) {
            json_stream << "\"name\": \"";
            json_stream << iter->util.m_bar_name;
            json_stream << "\"";
        }
        json_stream << "}";

        if (iter != traced_event_list.end() - 1) {
            json_stream << "},\n";
        } else {
            if (final == true) {
                json_stream << "}";
            } else {
                json_stream << "},\n";
            }
            // json_stream << "}";
        }
    }
}

// void Event_engine::writeEvents()
// {
// 	// collect all unique module names and give a pid
// 	typedef pair<string, unsigned> module_name_and_idx;
// 	map<string, unsigned> module_idx;
// 	unsigned pid_count = 1;
// 	for (auto iter = traced_event_list.begin(); iter !=
// traced_event_list.end(); iter++)
// 	{
// 		string m_name = iter->module_name;
// 		if (module_idx.end() == module_idx.find(m_name))
// 		{
// 			module_idx.insert(module_name_and_idx(m_name,
// pid_count)); 			pid_count++;
// 		}
// 	}

// 	for (auto iter = module_idx.begin(); iter != module_idx.end(); iter++)
// 	{
// 		json_stream << "{\"name\": \"process_name\", \"ph\": \"M\",
// \"pid\":"; 		json_stream << iter->second; 		json_stream <<
// ", \"args\":
// {\"name\": \""; 		json_stream << iter->first;
// json_stream << "\"}},\n";
// 	}

// 	// collect all thread names and give a tid
// 	typedef pair<string, string> module_name_and_thread_name;
// 	map<module_name_and_thread_name, unsigned> thread_idx;

// 	map<string, unsigned> thread_count_in_module = module_idx; // just for
// initialization 	for (auto iter = thread_count_in_module.begin(); iter !=
// thread_count_in_module.end(); iter++)
// 	{
// 		iter->second = 0;
// 	}

// 	for (auto iter = traced_event_list.begin(); iter !=
// traced_event_list.end(); iter++)
// 	{
// 		string m_name = iter->module_name;
// 		string t_name = iter->thread_name;
// 		module_name_and_thread_name m_t_name(m_name, t_name);
// 		if (thread_idx.end() == thread_idx.find(m_t_name))
// 		{
// 			thread_idx.insert(pair<module_name_and_thread_name,
// unsigned>(m_t_name, thread_count_in_module.at(m_name)));
// 			(thread_count_in_module.at(m_name)) ++;
// 		}
// 	}

// 	for (auto iter = thread_idx.begin(); iter != thread_idx.end(); iter++)
// 	{
// 		string m_name = iter->first.first;
// 		string t_name = iter->first.second;
// 		unsigned tid = iter->second;
// 		json_stream << "{\"name\": \"thread_name\", \"ph\": \"M\",
// \"pid\": "; 		json_stream << module_idx.at(m_name);
// json_stream << ", \"tid\":
// "; 		json_stream << thread_idx.at(module_name_and_thread_name(m_name,
// t_name)); 		json_stream << ", \"args\": {\"name\": \"";
// json_stream << t_name; 		json_stream << "\"}},\n";
// 	}


// 	// // write event
// 	// for (auto iter = traced_event_list.begin(); iter !=
// traced_event_list.end(); iter++)
// 	// {
// 	// 	string m_name = iter->module_name;
// 	// 	string t_name = iter->thread_name;
// 	// 	string bar_name = iter->util.m_bar_name;
// 	// 	string event_type = iter->type;
// 	// 	string color_type = iter->util.m_color;
// 	// 	float count_value = iter->util.m_value;
// 	// 	double ts = iter->time.to_default_time_units();

// 	// 	module_name_and_thread_name m_t_name(m_name, t_name);

// 	// 	json_stream << "{\"name\": \"";
// 	// 	if (bar_name.compare("") && event_type.compare("C")!=0) {
// 	// 		json_stream << bar_name;
// 	// 	}
// 	// 	else {
// 	// 		json_stream << t_name;
// 	// 	}
// 	// 	json_stream << "\", \"cat\": \"";
// 	// 	json_stream << m_name;
// 	// 	if (color_type != "None")
// 	// 	{
// 	// 		json_stream << "\", \"cname\": \"";
// 	// 		json_stream << color_type;
// 	// 	}
// 	// 	json_stream << "\", \"ph\": \"";
// 	// 	json_stream << event_type;
// 	// 	json_stream << "\", \"ts\": ";
// 	// 	json_stream << ts;
// 	// 	json_stream << ", \"pid\": ";
// 	// 	json_stream << module_idx.at(m_name);
// 	// 	json_stream << ", \"tid\": ";
// 	// 	json_stream << thread_idx.at(module_name_and_thread_name(m_name,
// t_name));
// 	// 	json_stream << ", \"args\": {";
// 	// 	if (event_type.compare("C") != 0) {
// 	// 		if (bar_name.compare("")) {
// 	// 			json_stream << "\"name\": \"";
// 	// 			json_stream << bar_name;
// 	// 			json_stream << "\"}";
// 	// 		} else json_stream << "}";
// 	// 	}
// 	// 	else {
// 	// 		string value_name = (bar_name.compare("") != 0) ?
// bar_name : t_name;
// 	// 		json_stream << "\"" << value_name << "\": ";
// 	// 		json_stream << count_value;
// 	// 		json_stream << "}";
// 	// 	}

// 	// 	if (iter != traced_event_list.end() - 1)
// 	// 	{
// 	// 		json_stream << "},\n";
// 	// 	}
// 	// 	else {
// 	// 		json_stream << "}";
// 	// 	}
// 	// }

// 	for (auto iter = traced_event_list.begin(); iter !=
// traced_event_list.end(); iter++)
//     {
//         string m_name = iter->module_name;
//         string t_name = iter->thread_name;
//         string event_type = iter->type;
//         //double ts = iter->time.to_default_time_units();
// 		double ts = iter->time.to_seconds() * 1e6;

//         json_stream << "{\"name\": \"";
//         json_stream << ((iter->util.m_bar_name != "") ? iter->util.m_bar_name
//         : t_name); json_stream << "\", \"cat\": \""; json_stream << m_name;
//         if (iter->util.m_color != "None") {
//             json_stream << "\", \"cname\": \"";
//             json_stream << iter->util.m_color;
//         }
//         json_stream << "\", \"ph\": \"";
//         json_stream << event_type;
//         json_stream << "\", \"ts\": ";
//         json_stream << ts;
//         json_stream << ", \"pid\": ";
//         json_stream << module_idx.at(m_name);
//         json_stream << ", \"tid\": ";
//         json_stream << thread_idx.at(module_name_and_thread_name(m_name,
//         t_name));

//         // 针对 flow event 的特殊处理
//         if (event_type == "s" || event_type == "f") { // Start or Flow end
//             json_stream << ", \"id\": ";
//             json_stream << iter->flow_id;
//             if (event_type == "f" && !iter->bp.empty()) {
//                 json_stream << ", \"bp\": \"";
//                 json_stream << iter->bp;
//                 json_stream << "\"";
//             }
//         }

//         // 写入通用 args 字段
//         json_stream << ", \"args\": {";
//         if (event_type == "C") {
//             string value_name = (!iter->util.m_bar_name.empty()) ?
//             iter->util.m_bar_name : t_name; json_stream << "\"" << value_name
//             << "\": "; json_stream << iter->util.m_value;
//         } else if (!iter->util.m_bar_name.empty()) {
//             json_stream << "\"name\": \"";
//             json_stream << iter->util.m_bar_name;
//             json_stream << "\"";
//         }
//         json_stream << "}";

//         if (iter != traced_event_list.end() - 1) {
//             json_stream << "},\n";
//         } else {
//             json_stream << "}";
//         }
//     }


// }


Event_engine::~Event_engine() { this->dump_traced_file("events.json", true); }
