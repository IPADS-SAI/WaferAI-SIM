#include "systemc.h"

#include "common/memory.h"
#include "defs/global.h"
#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"
#include "utils/system_utils.h"

REGISTER_PRIM(Set_addr);

void Set_addr::printSelf() {}

void Set_addr::deserialize(vector<sc_bv<128>> segments) {
    auto buffer = segments[0];

    sram_addr = buffer.range(31, 8).to_uint64();
    datatype = (DATATYPE)buffer.range(33, 32).to_uint64();

    int read_label_cnt = 0;

    for (int i = 1; i < segments.size() - 1; i++) {
        auto buffer = segments[i];

        for (int pos = 0; pos + 31 < 128 && read_label_cnt < MAX_SPLIT_NUM; pos += 32, read_label_cnt++) {
            datapass_label.indata[read_label_cnt] =
                g_addr_label_table.findRecord(buffer.range(pos + 31, pos).to_uint64());
        }
    }

    buffer = segments[segments.size() - 1];
    datapass_label.outdata = g_addr_label_table.findRecord(
        buffer.range(31, 0).to_uint64());
}

vector<sc_bv<128>> Set_addr::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> metadata;
    metadata.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    metadata.range(31, 8) = sc_bv<24>(sram_addr);
    metadata.range(33, 32) = sc_bv<2>(datatype);
    segments.push_back(metadata);

    int label_idx = 0;
    while (label_idx < MAX_SPLIT_NUM) {
        sc_bv<128> d;
        int pos = 0;
        for (; pos + 31 < 128 && label_idx < MAX_SPLIT_NUM; pos += 32, label_idx++) {
            d.range(pos + 31, pos) =
                sc_bv<32>(g_addr_label_table.addRecord(
                    prim_context->datapass_label_->indata[label_idx]));
        }
        segments.push_back(d);
    }

    sc_bv<128> d;
    d.range(31, 0) = sc_bv<32>(
        g_addr_label_table.addRecord(prim_context->datapass_label_->outdata));
    segments.push_back(d);

    return segments;
}

int Set_addr::taskCoreDefault(TaskCoreContext &context) {
    //  将datapass_label的内容复制到target中
    for (int i = 0; i < MAX_SPLIT_NUM; i++) {
        prim_context->datapass_label_->indata[i] = datapass_label.indata[i];
    }
    prim_context->datapass_label_->outdata = datapass_label.outdata;

    return 0;
}