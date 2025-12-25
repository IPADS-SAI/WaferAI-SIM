#include "systemc.h"

#include "prims/base.h"
#include "prims/norm_prims.h"
#include "utils/prim_utils.h"

REGISTER_PRIM(Load_prim);

void Load_prim::printSelf() {  }

void Load_prim::deserialize(vector<sc_bv<128>> segments) {
    auto buffer = segments[0];
}

vector<sc_bv<128>> Load_prim::serialize() {
    vector<sc_bv<128>> segments;

    sc_bv<128> d;
    d.range(7, 0) = sc_bv<8>(PrimFactory::getInstance().getPrimId(name));
    segments.push_back(d);

    return segments;
}
int Load_prim::taskCoreDefault(TaskCoreContext &context) { return 0; }