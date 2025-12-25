#pragma once
#include "systemc.h"

#include "prims/base.h"

class Matmul_f_gpu : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    Matmul_f_gpu() {
        name = "Matmul_f_gpu";
        param_name.insert(param_name.end(), {"B", "T", "C", "OC"});
    }
};

class Attention_f_gpu : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    Attention_f_gpu() {
        name = "Attention_f_gpu";
        param_name.insert(param_name.end(), {"B", "T", "C", "NH"});
    }
};

class Gelu_f_gpu : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    Gelu_f_gpu() {
        name = "Gelu_f_gpu";
        param_name.insert(param_name.end(), {"N"});
    }
};

class Layernorm_f_gpu : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    Layernorm_f_gpu() {
        name = "Layernorm_f_gpu";
        param_name.insert(param_name.end(), {"B", "T", "C"});
    }
};


class Residual_f_gpu : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    Residual_f_gpu() {
        name = "Residual_f_gpu";
        param_name.insert(param_name.end(), {"N"});
    }
};


class matmul_forward_gpu_pd : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    matmul_forward_gpu_pd() {
        name = "matmul_forward_gpu_pd";
        param_name.insert(param_name.end(),
                          {"B", "T", "C", "NH", "DH", "R", "OC", "job_type"});
    }
};

class attention_forward_gpu_pd : public GpuBase {
public:
    int taskCoreDefault(TaskCoreContext &context);
    void initialize();

    GpuBase *clone();

    attention_forward_gpu_pd() {
        name = "attention_forward_gpu_pd";
        param_name.insert(param_name.end(), {"B", "T", "C", "NH"});
    }
};