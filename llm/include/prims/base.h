#pragma once
#include "common/system.h"
#include "defs/const.h"
#include "defs/enums.h"
#include "defs/global.h"
#include "systemc.h"

#include "nlohmann/json.hpp"
#include <memory>
#include <vector>

using json = nlohmann::json;

class PrimBase {
public:
    shared_ptr<PrimCoreContext> prim_context;

    int prim_type;
    string name;

    int sram_addr;
    DATATYPE datatype = INT8;

    virtual int taskCoreDefault(TaskCoreContext &context) = 0;
    virtual vector<sc_bv<128>> serialize() = 0;
    virtual void deserialize(vector<sc_bv<128>>) = 0;
    virtual void printSelf() = 0;

    PrimBase() {
        name = "PrimBase";
        prim_type = NORM_PRIM;
        prim_context = nullptr;
    }
    virtual ~PrimBase() = default;
};


class CompBase : public PrimBase {
public:
    // 数据块信息
    vector<int> data_size_input;          // 必填，在initialize()中
    vector<pair<string, int>> data_chunk; // 必填，在initialize()中
    int input_size; // 可以推算，在initializeDefault()中
    int out_size;   // 可以推算，initializeDefault()中

    // 参数信息
    vector<string> param_name; // 必填，在构造函数中
    unordered_map<string, int>
        param_value; // 可以推算，在json中或在deserialize()中

    // 在initializeDefault()中
    int data_byte;

    virtual void initialize() = 0; // 解析之后进行的初始化函数，由用户自定义
    virtual void initializeDefault() = 0; // 默认初始化函数

    // 向上暴露的工作函数。不同硬件逻辑不同
    virtual int taskCoreDefault(TaskCoreContext &context) = 0;

    // 从配置文件中解析原语的工具函数
    virtual vector<sc_bv<128>> serialize() = 0;
    virtual void deserialize(vector<sc_bv<128>> buffer) = 0;
    virtual void parseJson(json j) = 0;

    // 打印原语信息
    virtual void printSelf() = 0;

    CompBase() { prim_type |= COMP_PRIM; }
};


class NpuBase : public CompBase {
public:
    // 地址偏移信息
    int inp_offset;  // 必填，在json文件中
    int data_offset; // 选填，可以推算，在parseJson()中
    int out_offset;  // 选填，可以推算，在parseJson()中

    // 是否跳过taskCoreDefault中的输入输出流程
    bool skip_input = false;
    bool skip_output = false;

    // 数据块信息
    unordered_map<string, int>
        data_chunk_addr; // 可以推算，在initializeDefault()中

    // 向上暴露的工作函数
    int taskCoreDefault(TaskCoreContext &context);
    virtual void taskCore(TaskCoreContext &context, string prim_name,
                          u_int64_t &dram_time, u_int64_t &exu_ops,
                          u_int64_t &sfu_ops, u_int64_t &vec_ops) = 0;

    // 原语解析函数
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void parseJson(json j);

    // 打印原语信息
    void printSelf();

    // perf工具函数
    int sramUtilization(DATATYPE datatype, int cid = 0);

    // 内存存取函数
    void checkStaticData(TaskCoreContext &context, uint64_t &dram_time,
                         uint64_t label_global_addr, int data_size_label,
                         string label_name, bool use_pf = false);
    void checkStaticDataTile(TaskCoreContext &context, uint64_t &dram_time,
                             uint64_t label_global_addr, int data_size_label,
                             string label_name, bool use_pf = false,
                             int mac_size = 128);
    void prefReadData(TaskCoreContext &context, uint64_t &dram_time,
                      int data_size_label, string label_name);

    NpuBase() { prim_type |= NPU_PRIM; }

private:
    // taskCore中的内存操作
    void checkInputData(TaskCoreContext &context, uint64_t &dram_time,
                        uint64_t inp_global_addr, vector<int> data_size_input);
    void writeOutputData(TaskCoreContext &context, uint64_t exu_flops,
                         uint64_t sfu_flops, u_int64_t vec_flops,
                         uint64_t dram_time, uint64_t &overlap_time,
                         int data_size_out, uint64_t out_global_addr);

    void parseAddress(json j);   // dram 地址
    void parseSramLabel(json j); // sram 标签

    // 初始化工具函数
    void initializeDefault();
};


class GpuBase : public CompBase {
public:
    // 原语的算力需求
    int req_sm;

    int fetch_index; // 用于记录取权重需要偏移的offset

    virtual GpuBase *clone() = 0;
    void initializeDefault();

    // 原语解析函数
    vector<sc_bv<128>> serialize();
    void deserialize(vector<sc_bv<128>> buffer);
    void parseJson(json j);

    // 打印原语信息
    void printSelf();

    GpuBase() {
        prim_type |= GPU_PRIM;
        param_name.insert(param_name.end(), {"slice_x", "slice_y"});
    }

private:
    void parseCompose(json j);
    void parseAddress(json j);
};


class PdBase : public NpuBase {
public:
    PdBase() {
        prim_type |= PD_PRIM;
        param_name.insert(param_name.end(), {"job_type"});
    }
};


class MoeBase : public NpuBase {
public:
    MoeBase() {
        prim_type |= MOE_PRIM;
        param_name.insert(param_name.end(), {"need_choose"});
    }
};