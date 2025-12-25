#include "prims/comp_prims.h"
#include "utils/system_utils.h"

REGISTER_PRIM(parse_input);

void parse_input::initialize() {
    auto &p = param_value;
    data_size_input = {p["size"]};
    data_chunk = {{"output", 0}};
}

void parse_input::taskCore(TaskCoreContext &context, string prim_name,
                           u_int64_t &dram_time, u_int64_t &exu_ops,
                           u_int64_t &sfu_ops, u_int64_t &vec_ops) {
    // 将input_label这个标签存储在指定label中
    string inp_label = INPUT_LABEL;
    prim_context->sram_pos_locator_->changePairName(
        inp_label, prim_context->datapass_label_->indata[0]);

    LOG_DEBUG(PRIM) << name << " of Core " << prim_context->cid
                    << " change input label to "
                    << prim_context->datapass_label_->indata[0];

    exu_ops = 0;
    sfu_ops = 0;
    vec_ops = 0;
}