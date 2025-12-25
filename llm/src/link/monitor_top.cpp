#include "link/monitor_top.h"
#include "link/config_top.h"
#include "link/monitor_chip.h"
#include "monitor/monitor.h"


TopMonitor::TopMonitor(sc_module_name name, Event_engine *event_engine,
                       TopConfig *config)
    : sc_module(name),
      config(config),
      event_engine(event_engine) {
    init();
}

TopMonitor::TopMonitor(sc_module_name name, Event_engine *event_engine,
                       std::string workload_config)
    : sc_module(name), event_engine(event_engine) {
    config = new TopConfig(workload_config);
    // TopMonitor(name, config);
    init();
}

void TopMonitor::init() {
    // 初始化组件
    assert(config != nullptr && "config is nullptr");

    for (auto config_ptr : config->component_) {
        switch (config_ptr->getType()) {
        case BaseConfig::TYPE_CHIP:
            // BaseComponent *base_component = new ChipMonitor("chip_monitor",
            // event_engine, config_ptr);
            components.push_back(new ChipMonitor("chip_monitor", event_engine,
                                                 config_ptr));
            break;
        default:
            assert(0 && "not implemented yet");
            break;
        }
    }
}