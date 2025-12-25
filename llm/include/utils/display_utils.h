#pragma once
#include <fstream>
#include <iostream>
#include <set>
#include <unordered_map>

#include "common/display.h"
#include "macros/macros.h"

#include "nlohmann/json.hpp"
#include <cairo/cairo.h>

using json = nlohmann::json;

// 解析 JSON 配置文件
json parse_config(const string &filename);

// 提取核心信息并建立核心数据流图
unordered_map<int, Display::Core> extract_core_data(const json &config);

#if USE_SFML == 1
// 绘制带描边的箭头
void draw_arrow(sf::RenderTexture &renderTexture, float start_x, float start_y,
                float end_x, float end_y, sf::Color fill_color,
                sf::Color outline_color);
void visualize_data_flow(sf::RenderTexture &renderTexture,
                         const unordered_map<int, Display::Core> &cores,
                         const set<int> source_ids);
void plot_dataflow(string filename);
#endif

#if USE_CARIO == 1
// 绘制带描边的箭头
void draw_arrow(cairo_t *cr, double start_x, double start_y, double end_x,
                double end_y, double r1, double g1, double b1, double r2,
                double g2, double b2);
void visualize_data_flow(cairo_surface_t *surface,
                         const unordered_map<int, Display::Core> &cores,
                         const set<int> source_ids);

void plot_dataflow(string filename);
void plot_dataflow(unordered_map<int, Display::Core> cores, set<int> source_ids);
#endif