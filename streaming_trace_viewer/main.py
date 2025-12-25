# main.py
import json
import os
import asyncio
import subprocess
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import socketio

app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
events = []
last_mtime = 0.0
TRACE_FILE = "../build/events.json"

# 全局变量：当前运行的模拟器进程和任务
current_proc = None
current_task = None

# 在文件顶部，global variables 区域
log_enabled_for_sid = {}  # sid -> bool

app.mount("/static", StaticFiles(directory="static"), name="static")


class SimulationRequest(BaseModel):
    workload_config: str = "../llm/test/workload_config/gpt2_small/original.json"
    hardware_config: str = "../llm/test/hardware_config/core_4x4.json"


@app.on_event("startup")
async def startup():
    global events, last_mtime
    if os.path.exists(TRACE_FILE):
        try:
            last_mtime = os.path.getmtime(TRACE_FILE)
            with open(TRACE_FILE, 'r') as f:
                data = json.load(f)
                events = data.get("traceEvents", [])
        except Exception as e:
            print("Load error:", e)
            events = []
    asyncio.create_task(poll_trace())


import re

async def poll_trace():
    global events, last_mtime
    while True:
        try:
            if os.path.exists(TRACE_FILE):
                mtime = os.path.getmtime(TRACE_FILE)
                if mtime > last_mtime:
                    with open(TRACE_FILE, 'r', encoding='utf-8') as f:
                        content = f.read()

                    new_events = []
                    # 使用正则提取 traceEvents 数组内容（非贪婪匹配）
                    # 1. 尝试提取 traceEvents 数组内容（允许不闭合）
                    match = re.search(r'"traceEvents"\s*:\s*\[([\s\S]*)', content)
                    if match:
                        array_content = match.group(1).strip()
                        
                        # 2. 移除可能的尾部垃圾（如多余的逗号、不完整的对象）
                        # 例如：..., {"name": "incomplete"
                        # 我们从后往前找最后一个完整的 }
                        last_brace = array_content.rfind('}}')
                        if last_brace != -1:
                            array_content = array_content[:last_brace + 2]
                        else:
                            array_content = ""  # 没有完整对象
                        # print(f"Extracted traceEvents: {array_content}")
                        
                        # 3. 现在 array_content 是类似: {"a":1}, {"b":2}, {"c":3}
                        if array_content:
                            # 按 "}," 分割
                            parts = array_content.split('},')
                            objects = []
                            for i, part in enumerate(parts):
                                if i < len(parts) - 1:
                                    json_str = part + '}'  # 补全
                                else:
                                    json_str = part  # 最后一个已处理过，应是完整对象
                                # print(f"Processing json: {json_str}")
                                try:
                                    obj = json.loads(json_str)
                                    objects.append(obj)
                                except json.JSONDecodeError:
                                    assert(False)
                                    continue  # 跳过无效对象
                            new_events = objects
                    else:
                        # 如果没有匹配到 traceEvents，尝试 fallback：找所有 {...} 对象
                        # （适用于极早期写入）
                        objects = []
                        brace_count = 0
                        current = ""
                        in_string = False
                        escape_next = False
                        for char in content:
                            if escape_next:
                                current += char
                                escape_next = False
                                continue
                            if char == '\\':
                                current += char
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                            current += char
                            if not in_string and brace_count == 0 and char == '}':
                                try:
                                    obj = json.loads(current.strip())
                                    objects.append(obj)
                                    current = ""
                                except:
                                    current = ""
                                    brace_count = 0
                            if brace_count < 0:
                                current = ""
                                brace_count = 0
                        new_events = objects

                    # 增量更新
                    if len(new_events) > len(events):
                        delta = new_events[len(events):]
                        events[:] = new_events
                        last_mtime = mtime
                        print(f"Loaded {len(delta)} new events (total: {len(new_events)})")
                        await sio.emit("new_events", delta)

        except Exception as e:
            print("Poll error:", e)
        await asyncio.sleep(0.5)
        
@sio.event
async def log_visibility(sid, hidden):
    """前端通知日志是否隐藏"""
    log_enabled_for_sid[sid] = not hidden
    print(f"Log visibility for {sid}: {'enabled' if not hidden else 'disabled'}")

async def read_stream(stream, proc_sid):
    """异步读取子进程输出，只推送给需要日志的客户端"""
    loop = asyncio.get_event_loop()
    buffer = b""
    while True:
        chunk = await loop.run_in_executor(None, stream.read, 4096)
        if not chunk:
            break
        buffer += chunk
        lines = buffer.split(b'\n')
        buffer = lines[-1]
        for line in lines[:-1]:
            decoded = line.decode('utf-8', errors='replace').rstrip()
            # 只推送给 log_enabled 的客户端
            for sid, enabled in log_enabled_for_sid.items():
                if enabled:
                    await sio.emit("log_message", decoded, to=sid)
        if buffer:
            await asyncio.sleep(0.01)
    if buffer:
        decoded = buffer.decode('utf-8', errors='replace').rstrip()
        if decoded:
            for sid, enabled in log_enabled_for_sid.items():
                if enabled:
                    await sio.emit("log_message", decoded, to=sid)

@sio.event
async def disconnect(sid):
    """客户端断开时清理状态"""
    log_enabled_for_sid.pop(sid, None)
    print(f"Client {sid} disconnected")


@app.post("/clear-trace")
async def clear_trace():
    global events, last_mtime
    try:
        # 删除文件（如果存在）
        print("Deleting trace file...")
        if os.path.exists(TRACE_FILE):
            os.remove(TRACE_FILE)
        
        # 清空内存中的事件
        events.clear()
        last_mtime = 0.0

        # 通知所有连接的客户端清空 trace
        await sio.emit("clear_events")

        return {"status": "success", "message": "Trace cleared"}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/run-simulation")
async def run_simulation(request: SimulationRequest):
    global current_proc, current_task

    # 如果已有进程在运行，先终止（可选）
    if current_proc and current_proc.poll() is None:
        return {"error": "Simulation already running"}

    try:
        npusim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "npusim"))
        if not os.path.exists(npusim_path):
            return {"error": f"npusim not found at {npusim_path}"}

        cmd = [
            npusim_path,
            "--workload-config", request.workload_config,
            "--hardware-config", request.hardware_config
        ]

        # 启动子进程，捕获 stdout/stderr
        current_proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(npusim_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            bufsize=1,  # 行缓冲
            universal_newlines=False  # 保持 bytes，避免编码问题
        )

        # 获取当前连接的 sid（简化：广播给所有；或可记录启动者 sid）
        # 此处为简化，直接广播日志（适合单用户）
        # 若需多用户隔离，需在前端 connect 时记录 sid 并绑定任务

        async def log_task():
            try:
                await read_stream(current_proc.stdout, None)  # None 表示广播
                current_proc.wait()
                await sio.emit("log_message", "[Simulation finished]")
            except Exception as e:
                await sio.emit("log_message", f"[Error reading log: {e}]")
            finally:
                global current_task
                current_task = None

        current_task = asyncio.create_task(log_task())

        return {
            "status": "success",
            "message": "Simulation started",
            "pid": current_proc.pid
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@sio.event
async def connect(sid, environ):
    await sio.emit("init_events", events, to=sid)


app_asgi = socketio.ASGIApp(sio, other_asgi_app=app)