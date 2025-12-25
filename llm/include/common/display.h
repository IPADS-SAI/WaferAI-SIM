#pragma once
#include <vector>

using namespace std;

// 以下为绘图相关
namespace Display {
// 表示绘图中 Core 和它们的数据流
struct Core {
    int id;
    vector<vector<int>> dests; // 数据流目标核心
    int x, y;                  // 在 2D 网格中的位置
};
} // namespace Display