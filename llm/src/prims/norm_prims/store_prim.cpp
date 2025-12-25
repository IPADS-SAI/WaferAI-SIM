#include "systemc.h"

#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

REGISTER_PRIM(Store_prim);

void Store_prim::printSelf() {  }

void Store_prim::deserialize(vector<sc_bv<128>> segments) {
    auto buffer = segments[0];

    dram_addr = buffer.range(23, 8).to_uint64();
    sram_addr = buffer.range(39, 24).to_uint64();
    size = buffer.range(55, 40).to_uint64();
    datatype = (DATATYPE)buffer.range(57, 56).to_uint64();
}

vector<sc_bv<128>> Store_prim::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    d.range(23, 8) = sc_bv<16>(dram_addr);
    d.range(39, 24) = sc_bv<16>(sram_addr);
    d.range(55, 40) = sc_bv<16>(size);
    d.range(57, 56) = sc_bv<2>(datatype);
    segments.push_back(d);

    return segments;
}
int Store_prim::taskCoreDefault(TaskCoreContext &context) { return 0; }