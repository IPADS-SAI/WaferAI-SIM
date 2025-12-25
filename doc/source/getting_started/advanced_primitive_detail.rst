.. _advanced_primitive_detail:

进阶原语书写方法
===================

本页面将介绍 **NPU-SIM** 中的三个工具原语，在熟悉基本的原语书写方法后，可灵活运用以下三个原语进行更复杂计算流的配置。

switch_data
~~~~~~~~~~~~~

可以将一块输入数据的大小变为指定的输出大小。该原语会引入额外的存储开销，在此忽略不计。

    **参数**
        - ``IN`` 输入大小
        - ``OUT`` 输出大小

    **SRAM地址**
        - ``indata`` 输入标签
        - ``outdata`` 输出标签

以下的示例将大小为1024的输入数据变为512的输出数据，并存储在了一个名为 ``output_label`` 的新标签中。

.. admonition:: 示例
    :class: tip

    .. code-block:: json

        {
            "type": "switch_data",
            "IN": 1024,
            "OUT": 512,
            "sram_address": {
                "indata": "input_label",
                "outdata": "output_label"
            }
        }

parse_input
~~~~~~~~~~~~~~~

在多工作核之间的复杂连续通信中，前一次的 ``input_label`` 可能还未来得及被使用，就被后续的 ``input_label`` 覆盖。此时可通过此原语将 ``input_label`` 指代的数据块重命名，以备后续使用。

    **参数** 
        - ``size`` 输入数据块大小

    **SRAM地址**
        - ``indata`` 需要的标签名
        - ``outdata`` 保持与 ``indata`` 一致

.. admonition:: 示例
    :class: tip

    .. code-block:: json

        {
            "type": "parse_input",
            "size": 1024,
            "sram_address": {
                "indata": "new_input_label",
                "outdata": "new_input_label"
            }
        }

parse_output
~~~~~~~~~~~~~~~

在 **NPU-SIM** 中，一个 ``worklist`` 的输出大小等于其最后一个计算原语的输出大小。而该原语被视为一个计算原语，因此可以将它放在一个 ``worklist`` 的最后，从而调整一个 ``worklist`` 的输出大小。

    **参数** 
        - ``size`` 输出大小
    
    **SRAM地址**
        - ``indata`` 保持与 ``outdata`` 一致
        - ``outdata`` 需要作为输出数据的数据块标签，如果在输出完之后需要将该数据块删除，则在标签前加上 ``DEL_``。

.. admonition:: 示例
    :class: tip

    .. code-block:: json

        {
            "type": "parse_output",
            "size": 512,
            "sram_address": {
                "indata": "aforementioned_output_label",
                "outdata": "DEL_aforementioned_output_label"
            }
        }