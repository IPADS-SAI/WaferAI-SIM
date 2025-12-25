#pragma once

// 作用与SEND,RECV TYPE相同，仅仅起到区分作用
enum GLOBAL_SEND_TYPE {
    GLOBAL_SEND_ACK = 0,
    GLOBAL_SEND_REQ,
    GLOBAL_SEND_DATA,
    GLOBAL_SEND_DONE,
};

enum GLOBAL_RECV_TYPE {
    GLOBAL_RECV_ACK = 0,
    GLOBAL_RECV_REQ,
    GLOBAL_RECV_DATA,
    GLOBAL_RECV_DONE,
};

// send原语的类型
enum SEND_TYPE {
    // 向RECV核发送握手信号
    SEND_ACK = 0,
    // 向RECV核发送数据请求信号
    SEND_REQ,
    // 向RECV核发送数据
    SEND_DATA,
    // DEPRECATED
    SEND_SRAM,
    // 最后一个核执行完所有指令后 向HOST发送DONE信号
    SEND_DONE,
};

// recv原语的类型
enum RECV_TYPE {
    // 接受HOST下发的原语
    RECV_CONF = 0,
    // SEND CORE 接受 RECV CORE 的握手信号
    RECV_ACK,
    // DEPRECATED
    RECV_FLAG,
    // 接受SEND核发送过来的路由数据
    RECV_DATA,
    // DEPRECATED
    RECV_SRAM,
    // 接收HOST下发的WEIGHT数据
    RECV_WEIGHT,
    // 接收初始数据
    RECV_START,
};

// 路由方向
enum Directions {
    WEST = 0,
    EAST,
    NORTH,
    SOUTH,
    CENTER,
    DIRECTIONS,
    HOST,
};

// 消息（数据包）类型
enum MSG_TYPE {
    CONFIG = 0,
    DATA,
    REQUEST,
    ACK,
    DONE,
    S_DATA, // start data
    P_DATA,  // prepare data
    MSG_TYPE_NUM
};

// 数据类型
enum DATATYPE {
    INT8 = 0,
    FP16,
};

// 通过硬件组件模拟原语
enum HardwareConduct { SYSTOLIC_MATMUL, UNDEFINED };

// 原语config中执行完毕之后是否继续循环
enum LOOP_TYPE { FALSE, TRUE, BOTH };

// config中split原语类型
enum SPLIT_TYPE { SPLIT_TP, SPLIT_DP, SPLIT_HYBRID, NO_SPLIT };

// workercore正在运行哪一种原语类型
enum CORE_PRIM {
    PRIM_RECV = 0,
    PRIM_COMP,
    PRIM_SEND,
};

// 模拟模式
enum SIM_MODE {
    SIM_DATAFLOW = 0,
    SIM_GPU,
    SIM_PD,
    SIM_PDS,
    SIM_GPU_PD,
};

// PD阶段
enum PD_PHASE {
    PREFILL = 0,
    DECODE,
    UNTOUCHED,
    PD_DONE,
};

// 在PD模式下每一个核需要进行的任务类型
enum PD_JOB {
    JOB_PREFILL = 0,
    JOB_DECODE,
    JOB_BOTH,
    JOB_NONE,
};

// 原语类型
enum PRIM_TYPE {
    NORM_PRIM = 0,
    COMP_PRIM = 1,
    GPU_PRIM = 1 << 1,
    NPU_PRIM = 1 << 2,
    PD_PRIM = 1 << 3,
    MOE_PRIM = 1 << 4,
};

// 用于计算核硬件配置
enum Etype { MAC_Array };
enum Sftype { Linear };

// 用于moe专家加载策略
enum MOE_LOAD_STRATEGY {
    MOE_LOAD_STRATEGY_NONE = 0,
    MOE_LOAD_STRATEGY_RANDOM,
    MOE_LOAD_STRATEGY_HOT,
    MOE_LOAD_STRATEGY_BEST,
};