#include "nlohmann/json.hpp"
#include <iostream>
#include <map>
#include <queue>
#include <vector>

#include "common/msg.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "monitor/config_helper_base.h"

using json = nlohmann::json;

bool config_helper_base::judge_is_end_core(int i) {
    // 判断一个核是否是汇节点
    CoreConfig c = coreconfigs[i];
    for (auto work : c.worklist) {
        for (auto cast : work.cast) {
            if (cast.dest == -1)
                return true;
        }
    }

    return false;
}

bool config_helper_base::judge_is_end_work(CoreJob work) {
    for (auto cast : work.cast) {
        if (cast.dest == -1)
            return true;
    }

    return false;
}

// 下发权重数据
void config_helper_base::fill_queue_data(queue<Msg> *q) {
    // 根据上述的地址生成策略，核之间的中间结果在input区域较前的位置，而初始生成的data会放在input区域较后的位置
    // dram内容安排如下：（左侧offset更小）
    // | output area | input (last core's output) | input (get from host) |
    // 可以复用delta offset这个map
    for (auto config : coreconfigs) {
        int index = config.id / GRID_X;
        int pkg_index = 0;
        int core_prim_cnt = 0;

        if (!SPEC_FAST_WARMUP) {
            for (auto work : config.worklist) {
                core_prim_cnt += work.prims.size();

                for (auto prim : work.prims) {
                    CompBase *cp = (CompBase *)prim;

                    // input_size 是 输入 input的大小
                    int send_size = cp->input_size;
                    int send_size_in_bit = send_size * sizeof(float) * 8;
                    int pkg_num = (send_size_in_bit % M_D_DATA)
                                      ? (send_size_in_bit / M_D_DATA + 1)
                                      : (send_size_in_bit / M_D_DATA);
                    pkg_num = pkg_num % HW_NOC_PAYLOAD_PER_CYCLE
                                  ? pkg_num / HW_NOC_PAYLOAD_PER_CYCLE + 1
                                  : pkg_num / HW_NOC_PAYLOAD_PER_CYCLE;

                    if (SPEC_USE_BEHA_NOC) {
                        sc_bv<M_D_DATA> d(0x1);
                        int length = M_D_DATA;
                        Msg m =
                            Msg(false, MSG_TYPE::P_DATA, ++pkg_index, config.id,
                                0, (1 << M_D_TAG_ID) - 1, length, d);
                        m.source_ = GRID_SIZE;
                        m.roofline_packets_ = pkg_num;
                        q[index].push(m);
                    } else {
                        for (int j = 1; j <= pkg_num; j++) {
                            // CTODO: 拿到真正的数据
                            sc_bv<M_D_DATA> d(0x1);
                            int length = M_D_DATA;
                            bool is_end_packet = j == pkg_num;
                            if (is_end_packet) {
                                length =
                                    send_size * 8 - M_D_DATA * (pkg_num - 1);
                            }

                            Msg m = Msg(false, MSG_TYPE::P_DATA, j + pkg_index,
                                        config.id, M_D_DATA * (j - 1),
                                        (1 << M_D_TAG_ID) - 1, length, d);
                            m.source_ = GRID_SIZE;
                            q[index].push(m);
                        }

                        pkg_index += pkg_num;
                    }
                }
            }
        }

        // HOST DATA END 包
        sc_bv<128> d(0x1);
        // Msg(bool e, MSG_TYPE m, int seq, int des, int offset, int tag, int
        // length, sc_bv<128> d) : is_end(e), msg_type(m), seq_id(seq),
        // des(des), offset(offset), tag_id(tag), length(length), data(d) {} (1
        // << M_D_TAG_ID) - 1 已被弃用 P_DATA 包的 tag
        // 不会被用于router中的lock，默认最大tag_id (1 << 16) - 1 end 包的
        // offset 弃用
        Msg m = Msg(true, MSG_TYPE::P_DATA, pkg_index + 1, config.id,
                    (1 << 16) - 1, (1 << M_D_TAG_ID) - 1, 0, d);
        m.source_ = GRID_SIZE;
        m.roofline_packets_ = 1;
        q[index].push(m);
    }
}