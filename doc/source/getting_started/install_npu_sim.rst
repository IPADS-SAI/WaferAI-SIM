.. _install_npu_sim:

安装 NPU-SIM
============

NPU-SIM 是一个基于 **SystemC** 编写的轻量级众核仿真器，能够灵活适配多种众核模式（包括 **SIMD** 与 **DataFlow**），
并支持 **LLM Serving** 等场景的仿真。

环境要求
--------

- **操作系统**：Linux  
- **SystemC**：2.3.3  
- **CMake**：3.31.3  
- **G++**：9.4.0  

安装方式
--------

.. _build_from_dockerfile:

方法一：通过 Dockerfile 安装（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 Dockerfile 构建镜像，构建过程大约耗时 **3 分钟**。

.. code-block:: console

    docker build -t npu-sim:latest .

构建完成后，运行容器：

.. code-block:: console

    docker run -it npu-sim:latest

进入容器后，可在当前目录下找到可执行文件 **npusim**。

.. _build_from_source:

方法二：通过源码安装
~~~~~~~~~~~~~~~~~~~~~~~~

以下步骤展示如何从源码编译安装 NPU-SIM。

安装 SystemC
^^^^^^^^^^^^

.. code-block:: console

    wget https://github.com/accellera-official/systemc/archive/refs/tags/2.3.3.tar.gz
    tar -zxvf 2.3.3.tar.gz
    cd systemc-2.3.3/
    mkdir tmp && cd tmp
    ../configure --prefix=/path/to/install/systemc-2.3.3 CXXFLAGS="-std=c++17"
    sudo make -j8
    make install

.. note::

   若需要使用 GDB 调试 SystemC，可在配置时加上以下参数：

   .. code-block:: console

       ../configure --prefix=/path/to/install/systemc-2.3.3_debug \
                    --enable-debug CXXFLAGS="-std=c++17"

安装 CMake 3.31.3
^^^^^^^^^^^^^^^^^

.. code-block:: console

    # 从官网下载安装包（https://cmake.org/download/）
    wget https://cmake.org/files/v3.31/cmake-3.31.3-linux-x86_64.tar.gz
    tar -zxvf cmake-3.31.3-linux-x86_64.tar.gz

安装 JSON 库
^^^^^^^^^^^^

.. code-block:: console

    git clone --branch=v3.11.3 --single-branch --depth=1 https://github.com/nlohmann/json.git
    cd json
    mkdir build && cd build
    cmake ..
    make
    sudo make install

安装多媒体与显示库
^^^^^^^^^^^^^^^^^^

.. code-block:: console

    # 安装 SFML
    sudo apt-get install libsfml-dev

    # 可能依赖以下组件
    sudo apt-get install libsfml-audio2.3v5 libopenal1 libopenal-data

    # 安装 Cairo
    sudo apt install libcairo2-dev

    # 安装 X11（服务器环境常需）
    sudo apt install xorg

    # 安装字体（源码中已包含必要的 ttf 文件）
    sudo apt install ttf-mscorefonts-installer
    # 安装过程中请在弹出界面选择 OK

配置环境变量
^^^^^^^^^^^^

.. code-block:: console

    export SYSTEMC_HOME=/path/to/install/systemc-2.3.3/
    export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$LD_LIBRARY_PATH
    export CMAKE_HOME=/path/to/install/cmake-3.31.3-linux-x86_64/bin/
    export PATH=$CMAKE_HOME:$PATH

下载并编译 NPU-SIM
^^^^^^^^^^^^^^^^^^

.. code-block:: console

    git clone https://gitee.com/doulujiyao/npu-sim
    cd npu-sim
    mkdir build && cd build

    # 正常构建
    cmake -DBUILD_DEBUG_TARGETS=OFF ..

    # 开启调试模式
    cmake -DBUILD_DEBUG_TARGETS=ON ..

.. note::

   仅测试了 **SystemC 2.3.3** 版本，其他版本可能存在不兼容问题。
