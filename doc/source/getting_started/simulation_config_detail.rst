.. _simulation_config_detail:

仿真配置
==========

仿真配置用于描述 **NPU-SIM** 在仿真过程中使用的仿真参数，以 **JSON** 格式保存。它主要描述了 **NPU-SIM** 对不同组件的仿真方式、优化技术、控制台输出等参数。

我们提供了示例配置文件，其中包含对于所有可配置参数的记录。它位于以下目录中：

.. code-block:: console

    ${NPU_SIM_ROOT}/llm/test/simulation_config

有关仿真配置的详细字段说明见下。

配置字段与书写规范
------------------

字体文件
~~~~~~~~~~~~~~~~

ttf_file : string
^^^^^^^^^^^^^^^^^^^

字体文件相对于主程序入口文件的路径。


算子
~~~~~~~~~~~~~~~~

operand : dict
^^^^^^^^^^^^^^^^

    **use_perf_gemm : boolean**

    是否使用性能优化的GEMM算子。在模拟计算周期时，会得到更准确的结果。

    **load_static_as_tile : boolean**

    在向SRAM中加载数据时，是否将静态数据作为tile进行加载。此方法会显著减少SRAM用量，从而减少溢出次数。但会引入额外的DRAM访问开销。


内存
~~~~~~~~~~~

memory : dict
^^^^^^^^^^^^^^^

    **use_beha_sram : boolean**

    是否使用行为级SRAM仿真。此方法会大幅加快仿真现实速度，但会略微丢失仿真精度。

    **use_beha_dram : boolean**

    是否使用行为级DRAM仿真。此方法会大幅加快仿真现实速度，但会略微丢失仿真精度。

    **use_dramsys : boolean**

    TODO 

片上网络（Network on Chip）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

noc : dict
^^^^^^^^^^^^

    **use_beha_noc : boolean**

    是否使用行为级片上网络仿真。此方法会大幅加快仿真现实速度，但会略微丢失仿真精度。

    **router_pipe : boolean**

    是否开启路由器的流水线模式。此方法可以隐藏路由器从SRAM中读取数据的时间。

    **fast_warmup : boolean**

    是否跳过激活数据流图的初始数据分发。

    **send_recv_parallel : boolean**

    是否同时执行工作核的SEND_PRIM和RECV_PRIM。在开启 ``use_beha_noc`` 时，启用此参数将不会有显著效果。

GPU
~~~~~~~~

gpu : dict
^^^^^^^^^^^^^^

    **use_inner_mm : boolean**

    执行矩阵乘法时，是否使用内积。若不开启，则采用外积。

    **cache_log : boolean**

    TODO

    **dram_config_file : string**

    指定GPU的内存配置文件相对于主程序文件入口的路径。

控制台输出
~~~~~~~~~~~~

log : dict
^^^^^^^^^^^^^^

    **log_level : number**

    控制台输出的详细程度。0为输出所有信息，1为省略Debug信息。

    **verbose_debug : boolean**

    是否输出以 ``_debug`` 结尾的标签。关闭此参数可省略一部分的输出信息。

    **colored : boolean**

    是否在控制台输出中显示颜色。