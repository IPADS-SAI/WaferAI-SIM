#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

REGISTER_PRIM(Set_batch);

int Set_batch::taskCoreDefault(TaskCoreContext &context) {
    prim_context->loop_cnt++;
    prim_context->auto_pd_ = auto_pd;

    prim_context->batch_info_.clear();
    for (auto stage : batch_info) {
        if (auto_pd && prim_context->loop_cnt > auto_pd) {
            LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                            << " auto pd enabled, overriding stage info.";
            prim_context->batch_info_.push_back(
                Stage(prim_context->loop_cnt % auto_pd, PD_PHASE(DECODE), 1));
        } else if (auto_pd > 1)
            prim_context->batch_info_.push_back(Stage(
                prim_context->loop_cnt % auto_pd, stage.type, stage.token_num));
        else
            prim_context->batch_info_.push_back(Stage(stage.req_id, stage.type, stage.token_num));
    }

    return 0;
}

void Set_batch::printSelf() {}

void Set_batch::deserialize(vector<sc_bv<128>> segments) {
    // 解析metadata
    auto buffer = segments[0];
    int batch_size = buffer.range(23, 8).to_uint64();
    auto_pd = buffer.range(39, 24).to_uint64();

    for (int i = 1; i < segments.size(); i++) {
        auto buffer = segments[i];

        for (int pos = 0; pos + 21 < 128 && batch_info.size() < batch_size;
             pos += 22) {
            Stage s =
                Stage(buffer.range(pos + 7, pos).to_uint64(),
                      PD_PHASE(buffer.range(pos + 9, pos + 8).to_uint64()),
                      buffer.range(pos + 21, pos + 10).to_uint64());
            batch_info.push_back(s);
        }
    }
}

vector<sc_bv<128>> Set_batch::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> metadata;
    metadata.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(23, 8) = sc_bv<16>(batch_info.size());
    metadata.range(39, 24) = sc_bv<16>(auto_pd);
    segments.push_back(metadata);

    for (int i = 0; i < batch_info.size();) {
        sc_bv<128> d;
        int pos = 0;
        for (; pos + 21 < 128 && i < batch_info.size(); pos += 22, i++) {
            d.range(pos + 7, pos) = sc_bv<8>(batch_info[i].req_id);
            d.range(pos + 9, pos + 8) = sc_bv<2>(batch_info[i].type);
            d.range(pos + 21, pos + 10) = sc_bv<12>(batch_info[i].token_num);
        }
        segments.push_back(d);
    }

    return segments;
}