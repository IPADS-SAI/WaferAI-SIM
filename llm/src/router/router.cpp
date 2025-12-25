#include "router/router.h"
#include "utils/print_utils.h"

RouterMonitor::RouterMonitor(const sc_module_name &n,
                             Event_engine *event_engine)
    : sc_module(n), event_engine(event_engine) {
    // WEAK: we now assume there are only 1*GRID_X tiles
    routers = new RouterUnit *[GRID_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        routers[i] =
            new RouterUnit(sc_gen_unique_name("router"), i, this->event_engine);
    }
}

RouterMonitor::~RouterMonitor() {
    // free routers
    for (int i = 0; i < GRID_SIZE; i++) {
        delete (routers[i]);
    }
    delete (routers);
}

RouterUnit::RouterUnit(const sc_module_name &n, int rid,
                       Event_engine *event_engine)
    : sc_module(n), rid(rid), event_engine(event_engine) {
   
    host_data_sent_i = nullptr;
    host_data_sent_o = nullptr;
    host_channel_i = nullptr;
    host_channel_o = nullptr;
    host_buffer_i = nullptr;
    host_buffer_o = nullptr;
    host_ctrl_buffer_o = nullptr;  
    host_channel_avail_o = nullptr;
    host_ctrl_sent_o = nullptr;
    host_ctrl_channel_o = nullptr;
    // 初始全部通道均未上锁
    for (int i = 0; i < DIRECTIONS; i++) {
        input_lock[i] = 0;
        input_lock_ref[i] = 0;
        output_lock[i] = -1;
        output_lock_ref[i] = 0;
    }


    if (IsMarginCore(rid)) {
        host_buffer_i = new queue<sc_bv<256>>;
        host_buffer_o = new queue<sc_bv<256>>;       
        host_ctrl_buffer_o = new queue<sc_bv<256>>;  
        host_channel_i = new sc_in<sc_bv<256>>;
        host_channel_o = new sc_out<sc_bv<256>>;
        host_data_sent_i = new sc_in<bool>;
        host_data_sent_o = new sc_out<bool>;
        host_channel_avail_o = new sc_out<bool>;
        host_ctrl_sent_o = new sc_out<bool>;
        host_ctrl_channel_o = new sc_out<sc_bv<256>>;
    }

    SC_THREAD(trans_next_trigger);
    // 数据信道触发信号
    sensitive << data_sent_i[WEST].pos() << data_sent_i[EAST].pos()
              << data_sent_i[CENTER].pos() << data_sent_i[SOUTH].pos()
              << data_sent_i[NORTH].pos();
    if (IsMarginCore(rid))
        sensitive << host_data_sent_i->pos();
    sensitive << channel_avail_i[WEST].pos() << channel_avail_i[EAST].pos()
              << channel_avail_i[SOUTH].pos() << channel_avail_i[NORTH].pos();
    sensitive << core_busy_i.neg();
    // 控制信道触发信号
    sensitive << ctrl_sent_i[WEST].pos() << ctrl_sent_i[EAST].pos()
              << ctrl_sent_i[CENTER].pos() << ctrl_sent_i[SOUTH].pos()
              << ctrl_sent_i[NORTH].pos();
    sensitive << ctrl_channel_avail_i[WEST].pos() << ctrl_channel_avail_i[EAST].pos()
              << ctrl_channel_avail_i[SOUTH].pos() << ctrl_channel_avail_i[NORTH].pos();
    sensitive << ctrl_core_busy_i.neg();
    dont_initialize();

    SC_THREAD(router_execute);
    sensitive << need_next_trigger;
    dont_initialize();
}

void RouterUnit::end_of_elaboration() {
    // set signals
    for (int i = 0; i < DIRECTIONS; i++) {
        channel_avail_o[i].write(true);
        data_sent_o[i].write(false);
        ctrl_channel_avail_o[i].write(true);
        ctrl_sent_o[i].write(false);
    }

    if (IsMarginCore(rid)) {
        host_channel_avail_o->write(true);
        host_data_sent_o->write(false);
        host_ctrl_sent_o->write(false);
    }
}

void RouterUnit::router_execute() {
    while (true) {
        bool flag_trigger = false;

        // 将输出信号都设置为初始值false
        for (int i = 0; i < DIRECTIONS; i++) {
            channel_avail_o[i].write(false);
            data_sent_o[i].write(false);
            ctrl_channel_avail_o[i].write(false);
            ctrl_sent_o[i].write(false);
        }

        // ==================== 数据信道输入 ====================
        // [input] 4方向+cores - 数据信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (data_sent_i[i].read()) {
                // move the data into the buffer
                sc_bv<256> temp = channel_i[i].read();
                Msg tt = DeserializeMsg(temp);

                buffer_i[i].emplace(temp);

                // need trigger again
                flag_trigger = true;
            }
        }

        // ==================== 控制信道输入 ====================
        // [ctrl input] 4方向+cores - 控制信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (ctrl_sent_i[i].read()) {
                // move the control data into the ctrl buffer
                sc_bv<256> temp = ctrl_channel_i[i].read();
                Msg tt = DeserializeMsg(temp);

                ctrl_buffer_i[i].emplace(temp);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [input] host
        // if IsMarginCore
        if (host_buffer_i) {
            // host send data to core
            if (host_data_sent_i->read()) {
                // move the data into the buffer
                sc_bv<256> temp = host_channel_i->read();

                Msg tt = DeserializeMsg(temp);

                host_buffer_i->emplace(temp);

                // need trigger again
                flag_trigger = true;
            }
        }

        // ==================== 数据信道输出 ====================
        // [output] 4方向 - 数据信道
        for (int i = 0; i < DIRECTIONS - 1; i++) {
            // global update once
            data_sent_o[i].write(false);
            // 输出方向的buffer是否为满
            if (channel_avail_i[i].read() == false)
                continue;

            // shall not check when output buffer is empty
            // 对应输出的buffer非空
            if (!buffer_o[i].size())
                continue;

            sc_bv<256> temp = buffer_o[i].front();
            buffer_o[i].pop();

            Msg tt = DeserializeMsg(temp);

            channel_o[i].write(temp);
            data_sent_o[i].write(true);

            // need trigger again
            flag_trigger = true;
        }

        // ==================== 控制信道输出 ====================
        // [ctrl output] 4方向 - 控制信道
        for (int i = 0; i < DIRECTIONS - 1; i++) {
            // global update once
            ctrl_sent_o[i].write(false);
            // 控制信道输出方向的buffer是否为满
            if (ctrl_channel_avail_i[i].read() == false)
                continue;

            // 对应控制信道输出的buffer非空
            if (!ctrl_buffer_o[i].size())
                continue;

            sc_bv<256> temp = ctrl_buffer_o[i].front();
            ctrl_buffer_o[i].pop();

            Msg tt = DeserializeMsg(temp);

            ctrl_channel_o[i].write(temp);
            ctrl_sent_o[i].write(true);

            // need trigger again
            flag_trigger = true;
        }

        // [output] host - 数据信道
        // if IsMarginCore
        if (host_channel_i) {
            host_data_sent_o->write(false);
            // 输出到host方向上的buffer非空
            if (host_buffer_o->size()) {
                sc_bv<256> temp = host_buffer_o->front();
                host_buffer_o->pop();

                host_channel_o->write(temp);
                host_data_sent_o->write(true);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [ctrl output] host - 控制信道 
        // if IsMarginCore 控制消息发送到host
        if (host_ctrl_channel_o) {
            host_ctrl_sent_o->write(false);
            // 输出到host方向上的控制buffer非空
            if (host_ctrl_buffer_o->size()) {
                sc_bv<256> temp = host_ctrl_buffer_o->front();
                host_ctrl_buffer_o->pop();

                host_ctrl_channel_o->write(temp);
                host_ctrl_sent_o->write(true);

                // need trigger again
                flag_trigger = true;
            }
        }

        // [output] core - 数据信道
        // 输出到本地core内部的
        data_sent_o[CENTER].write(false);
        // 输出到本地core内的buffer非空
        if (buffer_o[CENTER].size()) {
            // core内部的接受队列是否满
            if (!core_busy_i.read()) {
                // move the data out of the buffer
                sc_bv<256> temp = buffer_o[CENTER].front();

                buffer_o[CENTER].pop();

                Msg tt = DeserializeMsg(temp);

                channel_o[CENTER].write(temp);
                data_sent_o[CENTER].write(true);
            }

            // need trigger again
            flag_trigger = true;
        }

        // [ctrl output] core - 控制信道
        // 输出控制消息到本地core
        ctrl_sent_o[CENTER].write(false);
        // 控制信道输出到本地core内的buffer非空
        if (ctrl_buffer_o[CENTER].size()) {
            // 控制信道的core接受队列是否满（独立于数据信道）
            if (!ctrl_core_busy_i.read()) {
                // move the ctrl data out of the buffer
                sc_bv<256> temp = ctrl_buffer_o[CENTER].front();

                ctrl_buffer_o[CENTER].pop();

                Msg tt = DeserializeMsg(temp);

                ctrl_channel_o[CENTER].write(temp);
                ctrl_sent_o[CENTER].write(true);
            }

            // need trigger again
            flag_trigger = true;
        }

        // [input -> output] host
        // host输入包 向 output 哪个方向输出
        if (host_channel_i && host_buffer_i->size()) {
            sc_bv<256> temp = host_buffer_i->front();
            int d = DeserializeMsg(temp).des_;
            // 先x后y的路由
            Directions next = GetNextHop(d, rid);

            if (buffer_o[next].size() < MAX_BUFFER_PACKET_SIZE &&
                output_lock[next] == -1) {
                host_buffer_i->pop();
                buffer_o[next].emplace(temp);

                flag_trigger = true;
            }
        }

        // ==================== 控制信道路由 ====================
        // [ctrl input -> ctrl output] 4方向+core - 控制信道路由
        for (int i = 0; i < DIRECTIONS; i++) {
            if (!ctrl_buffer_i[i].size())
                continue;

            sc_bv<256> temp = ctrl_buffer_i[i].front();
            Msg m = DeserializeMsg(temp);
            Directions out = GetNextHop(m.des_, rid);

            // 控制信道不需要上锁机制，直接检查buffer是否满
            // REQUEST包和其他控制消息（ACK/DONE）一样直接流动，不需要req_queue
            if (out == HOST) {
                if (host_ctrl_buffer_o->size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                ctrl_buffer_i[i].pop();
                host_ctrl_buffer_o->emplace(temp);
            } else {
                if (ctrl_buffer_o[out].size() >= MAX_BUFFER_PACKET_SIZE)
                    continue;
                ctrl_buffer_i[i].pop();
                ctrl_buffer_o[out].emplace(temp);
            }

            flag_trigger = true;
        }

        // ==================== 数据信道路由 ====================
        // FIX input -> output 的仲裁
        // [input -> output] 4方向+core - 数据信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (!buffer_i[i].size())
                continue;

            sc_bv<256> temp = buffer_i[i].front();
            Msg m = DeserializeMsg(temp);
            Directions out = GetNextHop(m.des_, rid);



            if (m.des_ != GRID_SIZE && output_lock[out] != -1 &&
                output_lock[out] !=
                    m.tag_id_) // 如果不发往host，且目标通道上锁，且目标上锁tag不等同于自己的tag：continue
                continue;
            if (out == HOST &&
                host_buffer_o->size() >=
                    MAX_BUFFER_PACKET_SIZE) // 如果发往host，但通道已满：continue
                continue;
            else if (
                out != HOST &&
                buffer_o[out].size() >=
                    MAX_BUFFER_PACKET_SIZE) // 如果不发往host，但通道已满：continue
                continue;


            // FIX 上锁应该在第一个DATA 包
            if (m.msg_type_ == DATA && m.seq_id_ == 1 && m.des_ != GRID_SIZE &&
                m.source_ != GRID_SIZE) {
                // i 是 ACK 的进入方向，需要计算 ACK 的输出方向
                if (output_lock[out] == -1) {
                    // 上锁
                    output_lock[out] = m.tag_id_;
                    output_lock_ref[out]++;

                    LOG_DEBUG(NETWORK)
                        << "Router " << rid << " set lock direction "
                        << GetEnumDirectionType(out);
                    LOG_DEBUG(NETWORK)
                        << "  lock tag " << output_lock[out]
                        << ", lock reference " << output_lock_ref[out];
                } else if (output_lock[out] == m.tag_id_) {
                    // 添加refcnt
                    // Two Ack 多发一 DATA 包 乱序 接受核的接受地址由 Send
                    // 包中地址决定
                    output_lock_ref[out]++;

                    LOG_DEBUG(NETWORK)
                        << "Router " << rid << " add lock reference "
                        << GetEnumDirectionType(out);
                    LOG_DEBUG(NETWORK)
                        << "  lock tag " << output_lock[out]
                        << ", lock reference " << output_lock_ref[out];
                } else {
                    // 并非对应tag，不予通过
                    continue;
                }
            }

            // [DATA] 最后一个数据包，需要减少refcnt，如果refcnt为0,则解锁
            // DTODO
            // 排除了Config DATA 包，不会减少 lock
            // START DATA 包也不会上锁？
            if (m.msg_type_ == DATA && m.is_end_ && m.source_ != GRID_SIZE &&
                m.des_ != GRID_SIZE) {
                // i 是 data 的进入方向，需要计算 data 的输出方向
                out = GetNextHop(m.des_, rid);

                output_lock_ref[out]--;

                LOG_DEBUG(NETWORK) << "Router " << rid << " unlock "
                                   << GetEnumDirectionType(out);
                LOG_DEBUG(NETWORK)
                    << "  lock tag " << output_lock[out]
                    << ", lock reference " << output_lock_ref[out];


                if (output_lock_ref[out] < 0) {
                    LOG_ERROR(NETWORK)
                        << "Router " << rid << " output reference below zero";
                } else if (output_lock_ref[out] == 0) {
                    output_lock[out] = -1;
                }
            }

            // 发送
            if (out == HOST) {
                buffer_i[i].pop();
                host_buffer_o->emplace(temp);

                flag_trigger = true;
            } else {
                buffer_i[i].pop();
                buffer_o[out].emplace(temp);

                flag_trigger = true;
            }
        }

        // [SIGNALS] 5方向 - 数据信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (buffer_i[i].size() < MAX_BUFFER_PACKET_SIZE) {
                channel_avail_o[i].write(true);
            } else {
                channel_avail_o[i].write(false);
            }
        }

        // [CTRL SIGNALS] 5方向 - 控制信道
        for (int i = 0; i < DIRECTIONS; i++) {
            if (ctrl_buffer_i[i].size() < MAX_BUFFER_PACKET_SIZE) {
                ctrl_channel_avail_o[i].write(true);
            } else {
                ctrl_channel_avail_o[i].write(false);
            }
        }

        // [SIGNALS] host
        if (host_channel_i) {
            if (host_buffer_i->size() < MAX_BUFFER_PACKET_SIZE) {
                host_channel_avail_o->write(true);
            } else {
                host_channel_avail_o->write(false);
            }
        }

        // trigger again
        if (flag_trigger)
            need_next_trigger.notify(CYCLE, SC_NS);

        wait();
    }
}

void RouterUnit::trans_next_trigger() {
    while (true) {
        // DAHU notify 0ns
        need_next_trigger.notify(CYCLE, SC_NS);
        wait();
    }
}

RouterUnit::~RouterUnit() {
    if (host_buffer_i) {
        delete host_buffer_i;
        delete host_buffer_o;
        delete host_ctrl_buffer_o; 
        delete host_channel_i;
        delete host_channel_o;
        delete host_data_sent_i;
        delete host_data_sent_o;
        delete host_channel_avail_o;
        delete host_ctrl_sent_o;
        delete host_ctrl_channel_o;
    }
}