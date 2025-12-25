#pragma once
#include "systemc.h"
#include "trace/Event_engine.h"
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

using namespace sc_core;
using namespace tlm;
using namespace tlm_utils;

class DummyDCache : public sc_module {
public:
    // 定义目标socket
    simple_target_socket<DummyDCache> target_socket;

    SC_CTOR(DummyDCache) : target_socket("target_socket") {
        // 注册b_transport回调函数
        target_socket.register_b_transport(this, &DummyDCache::b_transport);
    }

    // b_transport方法实现
    void b_transport(tlm::tlm_generic_payload &trans, sc_time &delay) {
        // 根据请求的地址和其他参数，执行简单的逻辑
        unsigned char *data_ptr = trans.get_data_ptr();
        unsigned int length = trans.get_data_length();
        unsigned int address = trans.get_address();

        // 这里可以实现一些简单的处理，假设是读取数据
        if (trans.is_read()) {
            // 模拟读取数据
            // for (unsigned int i = 0; i < length; i++) {
            //     data_ptr[i] = static_cast<unsigned char>(address + i); //
            //     假设数据为地址 + i
            // }
            trans.set_response_status(TLM_OK_RESPONSE); // 设置响应状态
        } else if (trans.is_write()) {
            // 模拟写数据（这里可以选择记录数据或忽略）
            std::cout << "Writing data to address 0x" << std::hex << address
                      << std::endl;
            trans.set_response_status(TLM_OK_RESPONSE); // 设置响应状态
        } else {
            trans.set_response_status(
                TLM_GENERIC_ERROR_RESPONSE); // 设置错误响应
        }

        // wait(delay); // 模拟延迟
    }
};