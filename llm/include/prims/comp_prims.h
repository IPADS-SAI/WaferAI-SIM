#pragma once
#include "systemc.h"

#include "common/system.h"
#include "prims/base.h"
#include "utils/prim_utils.h"

class Attention_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Attention_f() {
        name = "Attention_f";
        param_name.insert(param_name.end(), {"B", "T", "C", "NH", "R"});
    }
};


class Batchnorm_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Batchnorm_f() {
        name = "Batchnorm_f";
        param_name.insert(param_name.end(), {"B", "W", "H", "C"});
    }
};


class Conv_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Conv_f() {
        name = "Conv_f";
        param_name.insert(param_name.end(), {"B", "W", "H", "C", "pX", "pY",
                                             "sX", "sY", "kX", "kY", "F"});
    }
};


class Dummy_p : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Dummy_p() { name = "Dummy_p"; }
};

class gate_forward : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    gate_forward() {
        name = "gate_forward";
        param_name.insert(param_name.end(), {"B", "T", "C", "E_N", "K"});
    }
};

class Gelu_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Gelu_f() {
        name = "Gelu_f";
        param_name.insert(param_name.end(), {"N"});
    }
};


class Layernorm_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Layernorm_f() {
        name = "Layernorm_f";
        param_name.insert(param_name.end(), {"B", "T", "C"});
    }
};


class Matmul_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    Matmul_f() {
        name = "Matmul_f";
        param_name.insert(param_name.end(), {"B", "T", "C", "OC"});
    }
};


class Matmul_f_mla : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    Matmul_f_mla() {
        name = "Matmul_f_mla";
        param_name.insert(param_name.end(), {"B", "T", "C", "OC", "NH", "DH"});
    }
};


class switch_data : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    switch_data() {
        name = "switch_data";
        param_name.insert(param_name.end(), {"IN", "OUT"});
        skip_input = true;
    }
};


class Max_pool : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    Max_pool() {
        name = "Max_pool";
        param_name.insert(param_name.end(), {"B", "W", "H", "C", "pX", "pY",
                                             "sX", "sY", "kX", "kY"});
    }
};


class Merge_conv : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Merge_conv() {
        name = "Merge_conv";
        param_name.insert(param_name.end(), {"B", "T", "C", "dim", "slice"});
    }
};


class Merge_matmul : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Merge_matmul() {
        name = "Merge_matmul";
        param_name.insert(param_name.end(), {"B", "T", "C", "dim", "slice"});
    }
};


class Relu_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    Relu_f() {
        name = "Relu_f";
        param_name.insert(param_name.end(), {"N"});
    }
};


class Residual_f : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Residual_f() {
        name = "Residual_f";
        param_name.insert(param_name.end(), {"N"});
    }
};


class rmsnorm_forward : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    rmsnorm_forward() {
        name = "rmsnorm_forward";
        param_name.insert(param_name.end(), {"B", "T", "C"});
    }
};


class rope_forward : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    rope_forward() {
        name = "rope_forward";
        param_name.insert(param_name.end(), {"B", "T", "C", "NH"});
    }
};


class silu_forward : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    silu_forward() {
        name = "silu_forward";
        param_name.insert(param_name.end(), {"N"});
    }
};


class Split_conv : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Split_conv() {
        name = "Split_conv";
        param_name.insert(param_name.end(),
                          {"W", "H", "C", "B", "pX", "pY", "S", "K", "slice"});
    }
};


class Split_matmul : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Split_matmul() {
        name = "Split_matmul";
        param_name.insert(param_name.end(), {"B", "T", "C", "dim", "slice"});
    }
};


class swiglu_forward : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();
    swiglu_forward() {
        name = "swiglu_forward";
        param_name.insert(param_name.end(), {"N"});
    }
};


class Send_global_memory : public NpuBase {
public:
    int data_packet_id; // 已经发送的包数量

    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Send_global_memory() {
        name = "Send_global_memory";
        param_name.insert(param_name.end(),
                          {"type", "enable", "des_id", "des_offset",
                           "local_offset", "max_packet", "tag_id",
                           "end_length"});
    }
};

class Recv_global_memory : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    Recv_global_memory() {
        name = "Recv_global_memory";
        param_name.insert(param_name.end(), {"type", "tag_id", "recv_cnt"});
    }
};

class parse_input : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    parse_input() {
        name = "parse_input";
        param_name.insert(param_name.end(), {"size"});
        skip_input = true;
    }
};

class parse_output : public NpuBase {
public:
    void taskCore(TaskCoreContext &context, string prim_name,
                  u_int64_t &dram_time, u_int64_t &exu_ops, u_int64_t &sfu_ops,
                  u_int64_t &vec_ops);
    void initialize();

    parse_output() {
        name = "parse_output";
        param_name.insert(param_name.end(), {"size"});
        skip_input = true;
    }
};