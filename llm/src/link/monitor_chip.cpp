#include "link/monitor_chip.h"
#include "monitor/monitor.h"

ChipMonitor::ChipMonitor(const sc_module_name &name, Event_engine *event_engine,
                         BaseConfig *config)
    : sc_module(name),
      config(config),
      event_engine(event_engine) {
    init();
}

void ChipMonitor::init() {

    assert(config != nullptr && "chip config is nullptr");
    assert(config->getType() == BaseConfig::TYPE_CHIP &&
           "chip config is not a chip config");

    ChipConfig *chip_config = dynamic_cast<ChipConfig *>(config);
    Monitor *monitor = new Monitor("monitor", event_engine, chip_config->chip);
    // assert(0 && "not implemented yet");
}

void ChipMonitor::start_simu() { assert(0 && "not implemented yet"); }
