import subprocess, glob, os
from collections import deque

workloads = glob.glob("../llm/test/workload_config/paper/fig16/Stp1.json")
hardware = glob.glob("../llm/test/hardware_config/paper/fig16/*.json")
sim_cfg = glob.glob("../llm/test/simulation_config/paper/fig16/*.json")
map_cfg = glob.glob("../llm/test/mapping_config/default_mapping.spec")

output_file = "result_summary.txt"

with open(output_file, "w") as fout:
    for wl in workloads:
        for hw in hardware:
            for mmp in map_cfg:
                for sim in sim_cfg:
                    print(f"Running {wl}  x  {hw}  x  {mmp}  x  {sim}")

                    # 存最后三行
                    last_lines = deque(maxlen=3)

                    # 流式读取输出，不缓存全部内容
                    p = subprocess.Popen(
                        ["./npusim", "--workload-config", wl, "--hardware-config", hw, "--simulation-config", sim, "--mapping-config", mmp],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1 # 行缓冲
                    )

                    for line in p.stdout:
                        print(line, end="")   # 正常打印出来，不影响npusim输出体验
                        last_lines.append(line.strip())

                    p.wait()

                    if len(last_lines) >= 3:
                        useful = list(last_lines)[-3]
                    else:
                        useful = list(last_lines)[-1]

                    fout.write(f"{os.path.basename(wl)}, {os.path.basename(hw)}, {os.path.basename(sim)}, {useful}\n")
