.. _run_npu_sim:

运行 NPU-SIM
============

使用以下命令运行 NPU-SIM：

.. code-block:: console

    ${NPU_SIM_PATH} \
        --workload-config ${WORKLOAD_CONFIG_PATH} \
        --simulation-config ${SIMULATION_CONFIG_PATH} \
        --hardware-config ${HARDWARE_CONFIG_PATH} \
        --mapping-config ${MAPPING_CONFIG_PATH}

命令行参数说明
--------------

以下是各个命令行参数的说明：

- **--workload-config**  
  指定工作负载配置文件路径。  

  详情请参阅：:doc:`workload_config_detail`

- **--simulation-config**  
  指定仿真配置文件路径。  

  详情请参阅：:doc:`simulation_config_detail`

- **--hardware-config**  
  指定硬件配置文件路径。  

  详情请参阅：:doc:`hardware_config_detail`

- **--mapping-config**  
  指定映射配置文件路径。  

  详情请参阅：:doc:`mapping_config_detail`

使用图形化前端界面
--------------------

我们提供了图形化前端界面，在运行配置时可同步显示 tracing 文件，便于更好地了解各个算子的执行情况与核心之间的通信模式。

.. raw:: html

   <video width="640" height="360" controls>
     <source src="https://github.com/doulujiyao12/npu-sim/blob/master/doc/images/gui_video.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>


附属页面
--------

.. toctree::
   :maxdepth: 1

   workload_config_detail
   simulation_config_detail
   hardware_config_detail
   mapping_config_detail
