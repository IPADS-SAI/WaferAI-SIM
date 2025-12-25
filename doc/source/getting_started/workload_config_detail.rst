.. _workload_config_detail:

工作负载配置
============

工作负载配置用于定义 **NPU-SIM** 在仿真过程中所需的数据流图（Dataflow Graph），以 **JSON** 格式保存。它主要描述了模型结构、模型参数、计算原语（primitive）所对应的工作负载、以及各工作核之间的通信流程等内容。

我们提供了若干示例配置文件，与论文中的各类实验场景相对应。这些示例配置位于以下目录中：

.. code-block:: console

    ${NPU_SIM_ROOT}/llm/test/workload_config

详细的字段说明可参阅：:doc:`workload_config_syntax`

进阶原语配置说明可参阅： :doc:`advanced_primitive_detail`

生成配置
--------

方法一：使用 Python 脚本自动生成
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用官方提供的 **Python** 脚本，可以快速生成指定模型结构（如 **GPT**、**Qwen** 等）的工作负载配置。该方法支持自定义模型参数与并行策略，适用于常规模型的仿真任务，开箱即用。

需要注意的是，该方法 **不支持对模型结构进行自定义修改** 。如需了解使用方式，请参阅：:doc:`workload_config_script`

方法二：手动编写配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~

如果用户希望自定义模型结构，或专注于探索片上核间通信范式，可以选择 **手动编写工作负载配置文件** 。  

**NPU-SIM** 的工作负载配置文件采用一套可读性强、语义清晰的自定义语法，支持灵活描述算子与通信关系，便于研究与验证不同架构设计。

详细的配置语法说明，请参阅：:doc:`workload_config_syntax`

附属页面
--------

.. toctree::
   :maxdepth: 1

   workload_config_script
   workload_config_syntax
   workload_faq