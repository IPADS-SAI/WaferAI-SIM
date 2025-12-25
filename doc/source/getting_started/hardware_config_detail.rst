.. _hardware_config_detail:

硬件配置
========

硬件配置用于描述 **NPU-SIM** 所使用的硬件架构参数，以 **JSON** 格式保存。它主要描述了计算核的总个数、每个计算核的内存大小，带宽、计算核算力、片上网络的核间带宽等参数。

我们提供了若干示例配置文件，用以对不同场景进行仿真。此外硬件配置文件还支持异构计算核心的配置。具体配置可参考 :doc:`default_hw_config`。 这些示例配置位于以下目录中：

.. code-block:: console

    ${NPU_SIM_ROOT}/llm/test/hardware_config

硬件配置文件书写相较工作负载配置文件而言更加简单，详细的字段说明见下。

配置字段与书写规范
------------------

计算核阵列
~~~~~~~~~~~~~~~~

x : number
^^^^^^^^^^

计算核心排布为 ``x * x`` 的方阵。

片上网络（Network on Chip）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

noc : dict
^^^^^^^^^^^

片上网络的配置。

    **noc_payload_per_cycle : number**

    片上网络相邻计算核间信道，在一个时钟周期内可以传输的负载包个数（约定一个负载包为128个bits）。

算子
~~~~~~

operand : dict
^^^^^^^^^^^^^^^

运算算子的配置。

    **comp_util : double**

    算子的计算资源使用效率，即在计算与访存过程中的重叠时间比例。

    **core_credit : number**

    **适用模式：pd**

    在一拍中，一个计算核可以最多分配的计算任务量，与 ``pd_ratio`` 相关。

    **pd_ratio : number**

    **适用模式：pd**

    在 **pd** 模式中，规定一次Decode任务的计算任务量为1。此字段定义了一次Prefill任务的计算量是Decode任务的多少倍。

    .. admonition:: 示例
        :class: tip

        若 ``core_credit`` 为8， ``pd_ratio`` 为3，则一拍中可最多为一个计算核分配“2个Prefill + 2个Decode”或“1个Prefill + 5个Decode”或“0个Prefill + 8个Decode”的计算任务。

memory : dict
^^^^^^^^^^^^^^

片上内存的配置。

    **beha_dram_util : double**

    TODO

    **dram_default_bitwidth: number**

    DRAM的位宽。

    **sram_size: number**

    SRAM的大小，目前仅支持所有核心的SRAM大小相同。

gpu : dict
^^^^^^^^^^^

**适用模式：gpu, gpu_pd**

GPU的配置。

    **dram_bandwidth: number**

    DRAM的带宽。

    **dram_burst_size: number**

    TODO

    **dram_aligned : number**

    TODO

cores : dict[]
^^^^^^^^^^^^^^^^

计算核的配置。

    **id : number**

    计算核的编号。

    **exu_x : number**

    矩阵运算单元阵列的横向维度大小。

    **exu_y : number**

    矩阵运算单元阵列的纵向维度大小。

    **sfu_x : number**

    非线性运算单元的阵列大小。

    **sram_bitwidth : number**

    SRAM的位宽。

.. note::
    - ``core`` 数组的长度至少为1。

    - 若在 ``core`` 数组中省略任意核的配置（假设该核编号为 ``x`` ），则核 ``x`` 的配置将被自动设置等同于编号小于 ``x`` 的已定义核中编号最大的核配置。

附属页面
-------------

.. toctree::
   :maxdepth: 1

   default_hw_config