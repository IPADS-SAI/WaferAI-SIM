#pragma once
#include "defs/enums.h"

int GetInputSource(Directions dir, int pos);
bool IsMarginCore(int id);

Directions GetNextHop(int des, int pos);
Directions GetNextHopReverse(int des, int pos);
Directions GetOpposeDirection(Directions dir);