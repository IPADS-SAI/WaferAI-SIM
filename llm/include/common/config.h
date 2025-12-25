#pragma once
#include <vector>

#include "defs/enums.h"
#include "prims/base.h"


class ExuConfig {
public:
    Etype type; // exu type
    int x_dims; // exu x array
    int count; // systolic array count

    ExuConfig() : type(MAC_Array), x_dims(128), count(1) {}
    ExuConfig(Etype t, int x, int cnt) : type(t), x_dims(x), count(cnt) {}
};

class SfuConfig {
public:
    Sftype type; // exu type
    int x_dims;  // exu x array

    SfuConfig() : type(Linear), x_dims(16) {}
    SfuConfig(Sftype t, int x) : type(t), x_dims(x) {}
};

class VectorConfig {
public:
    int x_dims;
    int count;

    VectorConfig() : x_dims(64), count(1) {}
    VectorConfig(int x, int cnt) : x_dims(x), count(cnt) {}
};


class Cast {
public:
    int dest;
    int tag;
    int weight;
    int addr;
    bool critical; // 用于绘制数据流图
    LOOP_TYPE loopout;

    Cast() {}
    Cast(int dest) : dest(dest), tag(tag) {
        weight = 1;
        addr = -1;
        critical = false;
        loopout = BOTH;
    }
};

void from_json(const json &j, Cast &c);

class CoreJob {
public:
    vector<Cast> cast;

    int recv_cnt;
    int recv_tag;

    vector<PrimBase *> prims;
    vector<PrimBase *> prims_last_loop;
    vector<PrimBase *> prims_in_loop;

    void printSelf();
    CoreJob() {}
    CoreJob(int recv_cnt, int recv_tag, int loop)
        : recv_cnt(recv_cnt), recv_tag(recv_tag) {
        Cast new_cast;
        new_cast.dest = -1;
        cast.push_back(new_cast);
    }
};

void from_json(const json &j, CoreJob &c);


class CoreConfig {
public:
    int id;
    int prim_copy;       // 是否需要复制其他核的计算原语
    int send_global_mem; // 是否需要将计算结果发送给Global Mem

    int loop; // 用于指定所有CoreJob需要重复执行的次数

    vector<CoreJob> worklist;

    void printSelf();
};

void from_json(const json &j, CoreConfig &c);

class LayerConfig {
public:
    int id; // 全局的原语数组
    CompBase *prim;
    vector<Cast> cast;

    int loop;
    int repeat;

    SPLIT_TYPE split;
    int split_slice;
    int split_dim;
};

void from_json(const json &j, LayerConfig &c);

class StreamConfig {
public:
    int id;
    int loop;

    vector<PrimBase *> prims;
    vector<pair<string, int>> sources;
};

void from_json(const json &j, StreamConfig &c);

class CoreHWConfig {
public:
    int id;
    ExuConfig *exu;
    SfuConfig *sfu;
    VectorConfig *vec;

    string dram_config; // DRAM配置文件名
    int dram_bw;
    int sram_bitwidth; // SRAM的位宽

    void printSelf() {
        // cout << "CoreHWConfig: " << id << endl;
        // cout << "ExuConfig: " << exu->type << " " << exu->x_dims << " "
        //      << exu->x_dims << endl;
        // cout << "SfuConfig: " << sfu->type << " " << sfu->x_dims << endl;
        // cout << "DRAM Config: " << dram_config << " " << dram_bw << " "
        //      << sram_bitwidth << endl;
    }

    CoreHWConfig()
        : id(0),
          exu(nullptr),
          sfu(nullptr),
          vec(nullptr),
          dram_config(""),
          dram_bw(0),
          sram_bitwidth(0) {}
    CoreHWConfig(int id, ExuConfig *exu, SfuConfig *sfu, VectorConfig *vec, string dram_config,
                 int dram_bw, int sram_bitwidth)
        : id(id),
          exu(exu),
          sfu(sfu),
          vec(vec),
          dram_config(dram_config),
          dram_bw(dram_bw),
          sram_bitwidth(sram_bitwidth) {}
    ~CoreHWConfig() {
        delete exu;
        delete sfu;
        delete vec;
    }
};

void from_json(const json &j, CoreHWConfig &c);