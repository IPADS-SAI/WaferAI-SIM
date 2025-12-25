#pragma once
#include "nlohmann/json.hpp"
#include "systemc.h"

#include "../monitor/config_helper_base.h"
#include "config_link_base.h"
#include "config_node_base.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

template <int WIDTH> class NaiveLink : public LinkConfig {
public:
    NaiveLink() : LinkConfig(NAIVE) {}
    sc_vector<sc_vector<sc_in<sc_bv<WIDTH>>>> link_channel_in;
    sc_vector<sc_vector<sc_out<sc_bv<WIDTH>>>> link_channel_out;
    sc_vector<sc_vector<sc_signal<bool>>> link_channel_valid;
    sc_vector<sc_vector<sc_signal<bool>>> link_channel_ready;
};