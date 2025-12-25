#include <fstream>
#include <iostream>
#include <set>

#include "common/display.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include "utils/display_utils.h"
#include "utils/print_utils.h"

#include "nlohmann/json.hpp"
#include <cairo/cairo.h>

using json = nlohmann::json;

// 解析 JSON 配置文件
json parse_config(const string &filename) {
    ifstream workload_config(filename);
    if (!workload_config.is_open()) {
        LOG_ERROR(display_utils.cpp)
            << "Cannot open workload config file " << filename;
    }
    json config;
    workload_config >> config;
    return config;
}

// 提取核心信息并建立核心数据流图
unordered_map<int, Display::Core> extract_core_data(const json &config) {
    unordered_map<int, Display::Core> cores;

    // 设定网格的行列数
    int gridX = GRID_X;
    int gridY = GRID_Y;

    // 计算每个 core 的位置
    int core_id = 0;
    for (const auto &core_json : config["chips"][0]["cores"]) {
        Display::Core core;
        core.id = core_json["id"];
        core.x = core.id % gridX; // X 坐标
        core.y = core.id / gridX; // Y 坐标

        // 提取每个 core 的 dest 信息
        if (!core_json.contains("worklist")) {
            vector<int> temp_cast;
            for (const auto &cast : core_json["cast"]) {
                if (cast.contains("critical") && cast["critical"]) {
                    int d = cast["dest"];
                    temp_cast.push_back(1e5 + d);
                } else
                    temp_cast.push_back(cast["dest"]);
            }

            core.dests.push_back(temp_cast);
        } else {
            for (const auto &work : core_json["worklist"]) {
                vector<int> temp_cast;
                for (const auto &cast : work["cast"]) {
                    if (cast.contains("critical") && cast["critical"]) {
                        int d = cast["dest"];
                        temp_cast.push_back(1e5 + d);
                    } else
                        temp_cast.push_back(cast["dest"]);
                }

                core.dests.push_back(temp_cast);
            }
        }

        cores[core.id] = core;
        core_id++;
    }

    return cores;
}

#if USE_SFML == 1
// 绘制带描边的箭头
void draw_arrow(sf::RenderTexture &renderTexture, float start_x, float start_y,
                float end_x, float end_y, sf::Color fill_color,
                sf::Color outline_color) {
    // 箭头线条宽度
    float line_thickness = 5.0f * 5 / GRID_X;

    // 绘制箭头主体线条（宽度更粗）
    sf::RectangleShape line(
        sf::Vector2f(sqrt(pow(end_x - start_x, 2) + pow(end_y - start_y, 2)),
                     line_thickness));
    line.setFillColor(fill_color);
    line.setOutlineColor(outline_color); // 添加描边
    line.setOutlineThickness(2.0f * 5 / GRID_X);
    line.setOrigin(0, line_thickness / 2); // 设置原点为线的左中点
    line.setPosition(start_x, start_y);
    line.setRotation(atan2(end_y - start_y, end_x - start_x) * 180 /
                     M_PI); // 根据两点计算旋转角度
    renderTexture.draw(line);

    // 计算箭头头部的方向
    float angle = atan2(end_y - start_y, end_x - start_x);
    float arrow_length = 25.0f * 5 / GRID_X; // 箭头长度
    float arrow_angle = M_PI / 6;            // 箭头两侧的角度

    // 创建箭头头部的三角形
    sf::ConvexShape arrow_head;
    arrow_head.setPointCount(3);
    arrow_head.setPoint(0, sf::Vector2f(end_x, end_y)); // 箭头尖端
    arrow_head.setPoint(
        1, sf::Vector2f(end_x - arrow_length * cos(angle - arrow_angle),
                        end_y - arrow_length * sin(angle - arrow_angle)));
    arrow_head.setPoint(
        2, sf::Vector2f(end_x - arrow_length * cos(angle + arrow_angle),
                        end_y - arrow_length * sin(angle + arrow_angle)));
    arrow_head.setFillColor(fill_color);               // 箭头填充颜色
    arrow_head.setOutlineColor(outline_color);         // 箭头边框颜色
    arrow_head.setOutlineThickness(3.0f * 5 / GRID_X); // 箭头边框厚度

    renderTexture.draw(arrow_head);
}


void visualize_data_flow(sf::RenderTexture &renderTexture,
                         const unordered_map<int, Display::Core> &cores,
                         const set<int> source_ids) {
    // 加载字体
    sf::Font font;
    if (!font.loadFromFile(SPEC_TTF_FILE)) {
        cerr << "无法加载字体！" << endl;
        exit(1);
    }

    // 计算核心框的间隔和尺寸
    float core_width = 150 * 5.0 / GRID_X;  // 核心框宽度
    float core_height = 150 * 5.0 / GRID_X; // 核心框高度
    float spacing = 100 * 5.0 / GRID_X;     // 核心框间距
    float offset_x =
        300 * 5.0 / GRID_X; // 水平偏移量（为避免与“Mem Interface”重叠）
    float offset_y = 50 * 5.0 / GRID_X; // 垂直偏移量

    // 渲染范围（为了计算从左下角开始的布局）
    float render_height = renderTexture.getSize().y;

    // 提取 Source 核心 ID
    // set<int> source_ids;
    // for (const auto& source : config["source"]) {
    //     source_ids.emplace(source["dest"]);
    // }

    // 计算核心网格的高度（用于调整 Mem Interface 的高度）
    int max_y = GRID_SIZE / GRID_X - 1;
    float mem_interface_height = (max_y + 1) * (core_height + spacing) -
                                 spacing; // 计算 Mem Interface 的高度

    // 绘制“Mem Interface”长方体
    sf::RectangleShape mem_interface_box(
        sf::Vector2f(200 * 5 / GRID_X, mem_interface_height)); // 调整高度
    mem_interface_box.setFillColor(sf::Color(173, 216, 230));  // 浅蓝色
    mem_interface_box.setOutlineColor(sf::Color(0, 0, 139));   // 深蓝色
    mem_interface_box.setOutlineThickness(5);
    mem_interface_box.setPosition(50, render_height - mem_interface_height -
                                          offset_y); // 左侧绘制，顶部对齐核心框

    // // 绘制“Mem Interface”文字
    // sf::Text mem_text;
    // mem_text.setFont(font);
    // mem_text.setString("Mem Interface");
    // mem_text.setCharacterSize(36);
    // mem_text.setStyle(sf::Text::Bold);
    // mem_text.setFillColor(sf::Color::Black);
    // sf::FloatRect memBounds = mem_text.getLocalBounds();
    // mem_text.setOrigin(memBounds.width / 2, memBounds.height / 2);
    // mem_text.setPosition(150, render_height - mem_interface_height / 2 -
    // offset_y);  // 文字居中绘制在“Mem Interface”内部

    // renderTexture.draw(mem_interface_box);
    // renderTexture.draw(mem_text);
    // 绘制“Mem Interface”竖着的文字
    renderTexture.draw(mem_interface_box);
    std::string mem_text_string = "Mem Interface";
    float text_start_x = (150 * 5.0 / GRID_X - 150) / 2 +
                         150; // 文字 X 坐标（Mem Interface 的中心）
    float text_start_y = render_height - mem_interface_height / 2 - offset_y -
                         300;               // 文字开始绘制的 Y 坐标
    float text_spacing = 45 * 5.0 / GRID_X; // 每个字符之间的垂直间距

    for (size_t i = 0; i < mem_text_string.length(); ++i) {
        sf::Text mem_text;
        mem_text.setFont(font);
        mem_text.setString(mem_text_string[i]); // 当前字符
        mem_text.setCharacterSize(36 * 5.0 / GRID_X);
        mem_text.setStyle(sf::Text::Bold);
        mem_text.setFillColor(sf::Color::Black);

        // 设置文字的位置
        sf::FloatRect memBounds = mem_text.getLocalBounds();
        mem_text.setOrigin(memBounds.width / 2, memBounds.height / 2);
        mem_text.setPosition(text_start_x,
                             text_start_y + i * text_spacing); // 按垂直方向排列

        renderTexture.draw(mem_text);
    }

    // 第一步：绘制所有核心框和文字
    unordered_map<int, Display::Core> temp_cores;
    for (int i = 0; i < GRID_SIZE; i++) {
        auto core_i = cores.find(i);
        if (core_i != cores.end()) {
            temp_cores[i] = core_i->second;
        } else {
            Display::Core tcore;
            tcore.id = i;
            tcore.x = i % GRID_X;
            tcore.y = i / GRID_X;
            temp_cores[i] = tcore;
        }
    }

    for (const auto &pair : temp_cores) {
        const Core &core = pair.second;

        // 将坐标系统调整为以左下角为原点
        float pos_x =
            offset_x + core.x * (core_width +
                                 spacing); // 水平方向从 Mem Interface 右侧开始
        float pos_y = render_height - offset_y -
                      core.y * (core_height + spacing) - core_height;

        // 判断是否是 Source 或 Destination
        sf::Color fill_color =
            sf::Color(255, 228, 181); // 默认填充颜色（浅橙色）
        if (source_ids.find(core.id) != source_ids.end()) {
            fill_color = sf::Color(135, 206, 235); // Source 用浅蓝色
        } else {
            for (auto work : core.dests) {
                for (auto cast : work) {
                    if (cast == -1)
                        fill_color = sf::Color(
                            255, 182, 193); // Destination（无目标）用浅粉色
                }
            }
        }

        // 绘制圆角矩形代表核心
        sf::RectangleShape core_box(sf::Vector2f(core_width, core_height));
        core_box.setFillColor(
            fill_color); // 使用不同颜色区分 Source 和 Destination
        core_box.setOutlineColor(sf::Color(255, 140, 0)); // 边框颜色（深橙色）
        core_box.setOutlineThickness(5 * 5.0 / GRID_X);   // 边框厚度
        core_box.setPosition(pos_x, pos_y);               // 计算核心框位置
        renderTexture.draw(core_box);

        // 绘制核心 ID（文字）
        sf::Text text;
        text.setFont(font);
        text.setString("Core " + to_string(core.id));
        text.setCharacterSize(24 * 5.0 / GRID_X);  // 字体大小
        text.setFillColor(sf::Color(101, 67, 33)); // 棕色字体
        text.setStyle(sf::Text::Bold);             // 加粗文字
        sf::FloatRect textBounds = text.getLocalBounds();
        text.setOrigin(textBounds.width / 2,
                       textBounds.height / 2); // 设置原点为中心
        text.setPosition(pos_x + core_width / 2,
                         pos_y + core_height / 2); // 文字居中
        renderTexture.draw(text);

        // 绘制从“Mem Interface”到第一列的箭头
        if (source_ids.find(core.id) != source_ids.end()) { // 判断是否是 Source
            float mem_x = 250; // 箭头起点（“Mem Interface”右侧边缘）
            float mem_y = render_height - offset_y -
                          core.y * (core_height + spacing) - core_height / 2;
            float core_center_x = pos_x; // 箭头终点（核心框左中点）
            float core_center_y = pos_y + core_height / 2;

            draw_arrow(renderTexture, mem_x, mem_y,  // 起点
                       core_center_x, core_center_y, // 终点
                       sf::Color::Black,             // 箭头填充颜色（黑色）
                       sf::Color::Black);            // 箭头描边颜色（黑色）
        }
    }

    // 第二步：绘制所有数据流箭头（箭头绘制在核心框之上）
    for (const auto &pair : cores) {
        const Display::Core &core = pair.second;

        for (int w = 0; w < core.dests.size(); w++) {
            auto work = core.dests[w];
            for (int d = 0; d < work.size(); d++) {
                auto dest = work[d];

                bool critical = dest >= 1e5;
                if (critical)
                    dest -= 1e5;

                if (cores.find(dest) != cores.end()) {
                    const Display::Core &dest_core = cores.at(dest);

                    // 计算起点和终点在新坐标系中的位置
                    float start_x = offset_x + core.x * (core_width + spacing) +
                                    core_width / 2;
                    float start_y = render_height - offset_y -
                                    core.y * (core_height + spacing) -
                                    core_height / 2;
                    float end_x = offset_x +
                                  dest_core.x * (core_width + spacing) +
                                  core_width / 2;
                    float end_y = render_height - offset_y -
                                  dest_core.y * (core_height + spacing) -
                                  core_height / 2;

                    if (critical) {
                        draw_arrow(
                            renderTexture, start_x,
                            start_y,                // 起点：当前核心的中心
                            end_x, end_y,           // 终点：目标核心的中心
                            sf::Color(0, 255, 128), // 箭头填充颜色（绿色）
                            sf::Color(0, 153, 76)); // 箭头描边颜色（深绿色）
                    } else {
                        float dx = end_x - start_x;
                        float dy = end_y - start_y;
                        auto angle = std::atan2(dy, dx) * 180 / M_PI;
                        bool reverse = false;
                        if (angle >= 90.0 - 0.1 || angle < -90.0 - 0.1) {
                            reverse = true;
                            swap(end_x, start_x);
                            swap(end_y, start_y);
                        }

                        dx = end_x - start_x;
                        dy = end_y - start_y;
                        float length = std::sqrt(dx * dx + dy * dy);
                        float mx = dx / length;
                        float my = dy / length;
                        start_x += my * 30 + mx * 30;
                        start_y -= mx * 30 - my * 30;
                        end_x += my * 30 - mx * 30;
                        end_y -= mx * 30 + my * 30;

                        if (reverse) {
                            swap(end_x, start_x);
                            swap(end_y, start_y);
                        }

                        draw_arrow(
                            renderTexture, start_x,
                            start_y,                 // 起点：当前核心的中心
                            end_x, end_y,            // 终点：目标核心的中心
                            sf::Color(186, 85, 211), // 箭头填充颜色（紫色）
                            sf::Color(148, 0, 211)); // 箭头描边颜色（深紫色）
                    }
                }
            }
        }
    }
}


void plot_dataflow(string filename) {
    // sf::Context::Settings settings;
    // settings.attributeFlags = sf::Context::ATTRIBUTE_DEFAULT |
    // sf::Context::ATTRIBUTE_NON_CLIENT; sf::Context context(settings);
    // 解析配置文件
    string workload_config = filename; // 替换为你的 JSON 配置文件路径
    json config = parse_config(workload_config);

    // 提取核心数据流图
    unordered_map<int, Display::Core> cores = extract_core_data(config);

    set<int> source_ids;
    for (const auto &source : config["source"]) {
        source_ids.emplace(source["dest"]);
    }

    // 创建 SFML 渲染目标
    sf::RenderTexture renderTexture;
    if (!renderTexture.create(800 * 2, 600 * 2)) {
        cerr << "无法创建渲染目标！" << endl;
    }

    // 在渲染目标中绘制
    renderTexture.clear(sf::Color::White);
    visualize_data_flow(renderTexture, cores, source_ids);
    renderTexture.display();

    // 保存为 PNG 图像文件
    if (!renderTexture.getTexture().copyToImage().saveToFile(
            "core_data_flow.png")) {
        cerr << "保存文件失败！" << endl;
    }

    LOG_INFO(SYSTEM) << "Image saved as 'core_data_flow.png'";
}

void plot_dataflow(unordered_map<int, Display::Core> cores,
                   set<int> source_ids) {
    // sf::Context::Settings settings;
    // settings.attributeFlags = sf::Context::ATTRIBUTE_DEFAULT |
    // sf::Context::ATTRIBUTE_NON_CLIENT; sf::Context context(settings); 创建
    // SFML 渲染目标
    sf::RenderTexture renderTexture;
    if (!renderTexture.create(800 * 2, 600 * 2)) {
        cerr << "无法创建渲染目标！" << endl;
    }

    // 在渲染目标中绘制
    renderTexture.clear(sf::Color::White);
    visualize_data_flow(renderTexture, cores, source_ids);
    renderTexture.display();

    // 保存为 PNG 图像文件
    if (!renderTexture.getTexture().copyToImage().saveToFile(
            "core_data_flow.png")) {
        cerr << "保存文件失败！" << endl;
    }

    LOG_INFO(SYSTEM) << "Image saved as 'core_data_flow.png'";
}

#endif

#if USE_CARIO == 1

// 绘制带描边的箭头
void draw_arrow(cairo_t *cr, double start_x, double start_y, double end_x,
                double end_y, double r1, double g1, double b1, double r2,
                double g2, double b2) {
    // 箭头线条宽度
    double line_thickness = 5.0 * 5 / GRID_X;

    // 计算箭头方向和长度
    double dx = end_x - start_x;
    double dy = end_y - start_y;
    double angle = atan2(dy, dx);

    // 保存当前状态
    cairo_save(cr);

    // 绘制箭头主体线条（带描边）
    cairo_set_line_width(cr, line_thickness + 2.0 * 5 / GRID_X); // 描边宽度
    cairo_set_source_rgb(cr, r2, g2, b2);                        // 描边颜色
    cairo_move_to(cr, start_x, start_y);
    cairo_line_to(cr, end_x, end_y);
    cairo_stroke(cr);

    // 绘制箭头主体线条（内部填充）
    cairo_set_line_width(cr, line_thickness);
    cairo_set_source_rgb(cr, r1, g1, b1); // 填充颜色
    cairo_move_to(cr, start_x, start_y);
    cairo_line_to(cr, end_x, end_y);
    cairo_stroke(cr);

    // 绘制箭头头部
    double arrow_length = 25.0 * 5 / GRID_X;
    double arrow_angle = M_PI / 6;

    // 计算箭头头部的三个点
    double tip_x = end_x;
    double tip_y = end_y;
    double left_x = end_x - arrow_length * cos(angle - arrow_angle);
    double left_y = end_y - arrow_length * sin(angle - arrow_angle);
    double right_x = end_x - arrow_length * cos(angle + arrow_angle);
    double right_y = end_y - arrow_length * sin(angle + arrow_angle);

    // 绘制箭头头部（描边）
    cairo_set_line_width(cr, 3.0 * 5 / GRID_X);
    cairo_set_source_rgb(cr, r2, g2, b2); // 描边颜色
    cairo_move_to(cr, tip_x, tip_y);
    cairo_line_to(cr, left_x, left_y);
    cairo_line_to(cr, right_x, right_y);
    cairo_close_path(cr);
    cairo_stroke_preserve(cr);

    // 填充箭头头部
    cairo_set_source_rgb(cr, r1, g1, b1); // 填充颜色
    cairo_fill(cr);

    // 恢复状态
    cairo_restore(cr);
}

void visualize_data_flow(cairo_surface_t *surface,
                         const unordered_map<int, Display::Core> &cores,
                         const set<int> source_ids) {
    cairo_t *cr = cairo_create(surface);

    // 设置白色背景
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // 获取渲染表面的高度
    double render_height = cairo_image_surface_get_height(surface);

    // 计算核心框的间隔和尺寸
    double core_width = 150 * 5.0 / GRID_X;  // 核心框宽度
    double core_height = 150 * 5.0 / GRID_X; // 核心框高度
    double spacing = 100 * 5.0 / GRID_X;     // 核心框间距
    double offset_x =
        300 * 5.0 / GRID_X; // 水平偏移量（为避免与"Mem Interface"重叠）
    double offset_y = 50 * 5.0 / GRID_X; // 垂直偏移量

    // 计算核心网格的高度（用于调整 Mem Interface 的高度）
    int max_y = GRID_SIZE / GRID_X - 1;
    double mem_interface_height = (max_y + 1) * (core_height + spacing) -
                                  spacing; // 计算 Mem Interface 的高度

    // 绘制"Mem Interface"长方体
    cairo_set_source_rgb(cr, 173 / 255.0, 216 / 255.0, 230 / 255.0); // 浅蓝色
    cairo_rectangle(cr, 50, render_height - mem_interface_height - offset_y,
                    200 * 5 / GRID_X, mem_interface_height);
    cairo_fill_preserve(cr);
    cairo_set_source_rgb(cr, 0, 0, 139 / 255.0); // 深蓝色
    cairo_set_line_width(cr, 5);
    cairo_stroke(cr);

    // 绘制"Mem Interface"竖着的文字
    std::string mem_text_string = "Mem Interface";
    double text_start_x = (150 * 5.0 / GRID_X - 150) / 2 +
                          150; // 文字 X 坐标（Mem Interface 的中心）
    double text_start_y = render_height - mem_interface_height / 2 - offset_y -
                          300;               // 文字开始绘制的 Y 坐标
    double text_spacing = 45 * 5.0 / GRID_X; // 每个字符之间的垂直间距

    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 36 * 5.0 / GRID_X);
    cairo_set_source_rgb(cr, 0, 0, 0); // 黑色

    for (size_t i = 0; i < mem_text_string.length(); ++i) {
        cairo_text_extents_t extents;
        std::string char_str(1, mem_text_string[i]);
        cairo_text_extents(cr, char_str.c_str(), &extents);

        cairo_move_to(cr, text_start_x - extents.width / 2,
                      text_start_y + i * text_spacing + extents.height / 2);
        cairo_show_text(cr, char_str.c_str());
    }

    // 第一步：绘制所有核心框和文字
    unordered_map<int, Display::Core> temp_cores;
    for (int i = 0; i < GRID_SIZE; i++) {
        auto core_i = cores.find(i);
        if (core_i != cores.end()) {
            temp_cores[i] = core_i->second;
        } else {
            Display::Core tcore;
            tcore.id = i;
            tcore.x = i % GRID_X;
            tcore.y = i / GRID_X;
            temp_cores[i] = tcore;
        }
    }

    for (const auto &pair : temp_cores) {
        const Display::Core &core = pair.second;

        // 将坐标系统调整为以左下角为原点
        double pos_x =
            offset_x + core.x * (core_width +
                                 spacing); // 水平方向从 Mem Interface 右侧开始
        double pos_y = render_height - offset_y -
                       core.y * (core_height + spacing) - core_height;

        // 判断是否是 Source 或 Destination
        double r, g, b;
        r = 255 / 255.0;
        g = 228 / 255.0;
        b = 181 / 255.0; // 默认填充颜色（浅橙色）

        if (source_ids.find(core.id) != source_ids.end()) {
            r = 135 / 255.0;
            g = 206 / 255.0;
            b = 235 / 255.0; // Source 用浅蓝色
        } else {
            for (auto work : core.dests) {
                for (auto cast : work) {
                    if (cast == -1) {
                        r = 255 / 255.0;
                        g = 182 / 255.0;
                        b = 193 / 255.0; // Destination（无目标）用浅粉色
                        break;
                    }
                }
            }
        }

        // 绘制矩形代表核心
        cairo_set_source_rgb(cr, r, g, b);
        cairo_rectangle(cr, pos_x, pos_y, core_width, core_height);
        cairo_fill_preserve(cr);
        cairo_set_source_rgb(cr, 255 / 255.0, 140 / 255.0,
                             0); // 边框颜色（深橙色）
        cairo_set_line_width(cr, 5 * 5.0 / GRID_X);
        cairo_stroke(cr);

        // 绘制核心 ID（文字）
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL,
                               CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(cr, 24 * 5.0 / GRID_X);
        cairo_set_source_rgb(cr, 101 / 255.0, 67 / 255.0,
                             33 / 255.0); // 棕色字体

        std::string core_text = "Core " + to_string(core.id);
        cairo_text_extents_t extents;
        cairo_text_extents(cr, core_text.c_str(), &extents);

        cairo_move_to(cr, pos_x + core_width / 2 - extents.width / 2,
                      pos_y + core_height / 2 + extents.height / 2);
        cairo_show_text(cr, core_text.c_str());

        // 绘制从"Mem Interface"到第一列的箭头
        if (source_ids.find(core.id) != source_ids.end()) { // 判断是否是 Source
            double mem_x = 250; // 箭头起点（"Mem Interface"右侧边缘）
            double mem_y = render_height - offset_y -
                           core.y * (core_height + spacing) - core_height / 2;
            double core_center_x = pos_x; // 箭头终点（核心框左中点）
            double core_center_y = pos_y + core_height / 2;

            draw_arrow(cr, mem_x, mem_y, core_center_x, core_center_y, 0, 0,
                       0,        // 箭头填充颜色（黑色）
                       0, 0, 0); // 箭头描边颜色（黑色）
        }
    }

    // 第二步：绘制所有数据流箭头（箭头绘制在核心框之上）
    for (const auto &pair : cores) {
        const Display::Core &core = pair.second;

        for (int w = 0; w < core.dests.size(); w++) {
            auto work = core.dests[w];
            for (int d = 0; d < work.size(); d++) {
                auto dest = work[d];

                bool critical = dest >= 1e5;
                if (critical)
                    dest -= 1e5;

                if (cores.find(dest) != cores.end()) {
                    const Display::Core &dest_core = cores.at(dest);

                    // 计算起点和终点在新坐标系中的位置
                    double start_x = offset_x +
                                     core.x * (core_width + spacing) +
                                     core_width / 2;
                    double start_y = render_height - offset_y -
                                     core.y * (core_height + spacing) -
                                     core_height / 2;
                    double end_x = offset_x +
                                   dest_core.x * (core_width + spacing) +
                                   core_width / 2;
                    double end_y = render_height - offset_y -
                                   dest_core.y * (core_height + spacing) -
                                   core_height / 2;

                    if (critical) {
                        draw_arrow(cr, start_x, start_y, end_x, end_y, 0,
                                   255 / 255.0,
                                   128 / 255.0, // 箭头填充颜色（绿色）
                                   0, 153 / 255.0,
                                   76 / 255.0); // 箭头描边颜色（深绿色）
                    } else {
                        double dx = end_x - start_x;
                        double dy = end_y - start_y;
                        auto angle = std::atan2(dy, dx) * 180 / M_PI;
                        bool reverse = false;
                        if (angle >= 90.0 - 0.1 || angle < -90.0 - 0.1) {
                            reverse = true;
                            swap(end_x, start_x);
                            swap(end_y, start_y);
                        }

                        dx = end_x - start_x;
                        dy = end_y - start_y;
                        double length = std::sqrt(dx * dx + dy * dy);
                        double mx = dx / length;
                        double my = dy / length;
                        start_x += my * 30 + mx * 30;
                        start_y -= mx * 30 - my * 30;
                        end_x += my * 30 - mx * 30;
                        end_y -= mx * 30 + my * 30;

                        if (reverse) {
                            swap(end_x, start_x);
                            swap(end_y, start_y);
                        }

                        draw_arrow(cr, start_x, start_y, end_x, end_y,
                                   186 / 255.0, 85 / 255.0,
                                   211 / 255.0, // 箭头填充颜色（紫色）
                                   148 / 255.0, 0,
                                   211 / 255.0); // 箭头描边颜色（深紫色）
                    }
                }
            }
        }
    }

    cairo_destroy(cr);
}

void plot_dataflow(string filename) {
    // 解析配置文件
    string workload_config = filename; // 替换为你的 JSON 配置文件路径
    json config = parse_config(workload_config);

    // 提取核心数据流图
    unordered_map<int, Display::Core> cores = extract_core_data(config);

    set<int> source_ids;
    for (const auto &source : config["source"]) {
        source_ids.emplace(source["dest"]);
    }

    // 创建 Cairo 渲染表面
    int width = 800 * 2;
    int height = 700 * 2;
    cairo_surface_t *surface =
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t *cr = cairo_create(surface);

    // 设置白色背景
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // 在渲染表面中绘制
    visualize_data_flow(surface, cores, source_ids);

    // 保存为 PNG 图像文件
    cairo_surface_write_to_png(surface, "core_data_flow.png");

    // 清理资源
    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    LOG_INFO(SYSTEM) << "Image saved as 'core_data_flow.png'";
}

void plot_dataflow(unordered_map<int, Display::Core> cores,
                   set<int> source_ids) {
    // 创建 Cairo 渲染表面
    int width = 800 * 2;
    int height = 600 * 2;
    cairo_surface_t *surface =
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t *cr = cairo_create(surface);

    // 设置白色背景
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    // 在渲染表面中绘制
    visualize_data_flow(surface, cores, source_ids);

    // 保存为 PNG 图像文件
    cairo_surface_write_to_png(surface, "core_data_flow.png");

    // 清理资源
    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    LOG_INFO(SYSTEM) << "Image saved as 'core_data_flow.png'";
}
#endif