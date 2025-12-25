![](doc/images/npusim.svg)

# NPU-SIM

NPU-SIM is a lightweight, large-scale, and multi-level simulation framework designed for multi-core Neural Processing Units (NPUs). It supports both transaction-level and performance-model-based simulation, providing powerful system-level analysis capabilities for large-scale models, such as Large Language Models (LLMs).

The framework is highly flexible and extensible, allowing for detailed simulation of various hardware and model configurations. Key features include:

* **Flexible Parallelism:** Exploration of various tensor parallelism strategies.
* **Customizable Core Placement:** Support for user-defined core placement policies.
* **Advanced Memory Management:** Simulation of diverse memory management methods.
* **Configurable Dataflow:** Selection between PD-disaggregation and PD-fusion on multi-core NPUs.

Furthermore, NPU-SIM extends its capabilities to the modeling of next-generation hardware by supporting wafer-scale simulation, enabling the analysis of systems that utilize hybrid bonding and distributed memory architectures

![](doc/images/arch.png)

## GUI Visualization

NPU-SIM provides an interactive GUI for real-time visualization of the simulation process:

 **[🎬 Demo Video:](./doc/images/gui_video.mp4)**
 
https://github.com/user-attachments/assets/16d4f604-30a7-43bc-a2e8-a6a776718708

## **Documentation:**  

📘 **Documentation:**  [Click here to access the NPU-SIM documentation.](https://npu-sim.readthedocs.io/zh-cn/latest/)


## Using NPU-SIM

### 1.Dependencies

- OS: Linux

- SystemC: 2.3.3

- Cmake: 3.31.3

- G++: 9.4.0



### 1.1 Installing SystemC

```bash
wget https://github.com/accellera-official/systemc/archive/refs/tags/2.3.3.tar.gz
tar -zxvf 2.3.3.tar.gz
cd systemc-2.3.3/
mkdir tmp && cd tmp
../configure --prefix=/path/to/install/systemc-2.3.3 CXXFLAGS="-std=c++17"
sudo make -j8
make install
```

Add the following to your `~/.bashrc` file:

```bash
export SYSTEMC_HOME=/path/to/install/systemc-2.3.3/
export LD_LIBRARY_PATH=/path/to/install/systemc-2.3.3/lib-linux64/:$LD_LIBRARY_PATH
```

### 1.2 Installing CMake 3.31.3
```bash
#https://cmake.org/download/ Download the source code or corresponding binary file from the Cmake official website
wget https://cmake.org/files/v3.31/cmake-3.31.3-linux-x86_64.tar.gz
tar -zxvf cmake-3.31.3-linux-x86_64.tar.gz
```

### 1.3 Installing the JSON library

```bash
git clone --branch=v3.11.3 --single-branch --depth=1 https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake ..
make
sudo make install
```

### 1.4 Installing the Multimedia Library

```bash
sudo apt-get install libsfml-dev
sudo apt install libcairo2-dev
sudo apt install xorg
sudo apt install ttf-mscorefonts-installer  
```

### 2. Compile and Run

```bash
cd /path/to/NPU-SIM/src
mkdir build && cd build
cmake ..
make -j8
```

```bash
 ./npusim \
    --workload-config ${WORKLOAD_CONFIG_PATH} \
    --simulation-config ${SIMULATION_CONFIG_PATH} \
    --hardware-config ${HARDWARE_CONFIG_PATH} \
     --mapping-config ${MAPPING_CONFIG_PATH}
```

```bash
 ./npusim \
    --workload-config ../llm/test/workload_config/gpu/pd_serving.json \
    --simulation-config ../llm/test/simulation_config/default_spec.json \
    --hardware-config ../llm/test/hardware_config/core_4x4.json \
    --mapping-config ../llm/test/mapping_config/default_mapping.txt
```
