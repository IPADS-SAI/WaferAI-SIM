#pragma once
#include "Trace_event.h"
#include <iostream>
#include <queue>
using namespace std;
class Trace_event_queue {
public:
    queue<Trace_event> trace_event_queue;
};
