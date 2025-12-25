#include "utils/router_utils.h"
#include "defs/global.h"
#include "defs/spec.h"
#include "macros/macros.h"
#include <queue>
#include <fstream>

int GetInputSource(Directions dir, int pos) {
    int x = pos % GRID_X;
    int y = pos / GRID_X;

    switch (dir) {
    case WEST:
        x = (x - 1 + GRID_X) % GRID_X;
        break;
    case NORTH:
        y = (y + 1) % GRID_X;
        break;
    case EAST:
        x = (x + 1) % GRID_X;
        break;
    case SOUTH:
        y = (y - 1 + GRID_X) % GRID_X;
        break;
    default:
        return 0;
    }

    return y * GRID_X + x;
}

bool IsMarginCore(int id) { return id % GRID_X == 0; }

Directions GetNextHop(int des, int pos) {
    // 从pos发往des的下一个方向, 先X后Y
    if (IsMarginCore(pos) && des == GRID_SIZE)
        return HOST;
    if (des == GRID_SIZE)
        return WEST; // CTODO: fix this

    int dx = des % GRID_X;
    int dy = des / GRID_X;

    int xx = pos % GRID_X;
    int yy = pos / GRID_X;

    if (dx == xx && dy == yy)
        return CENTER;
    else if (dx != xx) {
        if (dx > xx) {
            return EAST;
        } else {
            return WEST;
        }
    } else {
        if (dy > yy) {
            return NORTH;
        } else {
            return SOUTH;
        }
    }
}

Directions GetNextHopReverse(int des, int pos) {
    // 从pos发往des的下一个方向, 先Y后X
    if (IsMarginCore(pos) && des == GRID_SIZE)
        return HOST;
    if (des == GRID_SIZE)
        return WEST; // CTODO: fix this

    int dx = des % GRID_X;
    int dy = des / GRID_X;

    int xx = pos % GRID_X;
    int yy = pos / GRID_X;

    if (dx == xx && dy == yy)
        return CENTER;
    else if (dy != yy) {
        if (dy > yy) {
            return NORTH;
        } else {
            return SOUTH;
        }
    } else {
        if (dx > xx) {
            return EAST;
        } else {
            return WEST;
        }
    }
}

Directions GetOpposeDirection(Directions dir) {
    switch (dir) {
    case WEST:
        return EAST;
    case EAST:
        return WEST;
    case NORTH:
        return SOUTH;
    case SOUTH:
        return NORTH;
    default:
        return CENTER;
    }
}