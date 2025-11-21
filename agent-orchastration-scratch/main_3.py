
### # mcp_tool/server.py

# mcp_tool/server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import shlex
import subprocess
import os
import json

app = FastAPI(title="Mini-MCP Tool Server (streaming HTTP)")

# Tool registry: name -> metadata
TOOLS = {
    "calc_add": {
        "name": "calc_add",
        "description": "Add two numbers",
        "input_schema": {"a": "number", "b": "number"},
        "output_schema": {"result": "number"}
    },
    "run_shell": {
        "name": "run_shell",
        "description": "Run a whitelisted shell command",
        "input_schema": {"cmd": "string"},
        "output_schema": {"stdout": "string", "stderr": "string", "returncode": "int"}
    },
    "fake_search": {
        "name": "fake_search",
        "description": "Fake web search returning simulated results",
        "input_schema": {"query": "string"},
        "output_schema": {"results": "list"}
    }
}

ALLOWED_COMMANDS = {"echo", "uname", "date", "whoami", "ls"}  # whitelisted

class CallReq(BaseModel):
    args: Dict[str, Any] = {}

@app.get("/.well-known/tools")
async def list_tools():
    # Return tool metadata (discovery)
    return {"tools": list(TOOLS.values())}

@app.post("/call/{tool_name}")
async def call_tool(tool_name: str, req: CallReq):
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="tool_not_found")
    args = req.args or {}
    if tool_name == "calc_add":
        a = args.get("a"); b = args.get("b")
        if a is None or b is None:
            raise HTTPException(status_code=400, detail="a and b required")
        return {"status":"ok", "result": {"result": a + b}, "next_tasks": []}
    if tool_name == "fake_search":
        q = args.get("query","")
        # simulate async I/O
        await asyncio.sleep(0.2)
        return {"status":"ok", "result": {"results":[f"result for {q} #{i}" for i in range(1,4)]}, "next_tasks": []}
    if tool_name == "run_shell":
        cmd = args.get("cmd", "")
        if not cmd:
            raise HTTPException(status_code=400, detail="cmd required")
        parts = shlex.split(cmd)
        if not parts:
            raise HTTPException(status_code=400, detail="empty command")
        exe = os.path.basename(parts[0])
        if exe not in ALLOWED_COMMANDS:
            raise HTTPException(status_code=403, detail=f"command '{exe}' not allowed")
        # run command and capture output (blocking)
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return {"status":"ok", "result": {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}, "next_tasks": []}
    raise HTTPException(status_code=500, detail="no_impl")

# SSE streaming endpoint for long-running tools
@app.get("/stream/{tool_name}")
async def stream_tool(tool_name: str, request: Request):
    """
    Return a server-sent events stream. We simulate streaming output line-by-line.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="tool_not_found")

    async def event_generator():
        # For demo: stream three messages then finish
        for i in range(3):
            if await request.is_disconnected():
                break
            data = {"part": i+1, "message": f"stream chunk {i+1} from {tool_name}"}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.5)
        # end
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


mcp_tool/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY server.py /app/server.py
RUN pip install fastapi "uvicorn[standard]" pydantic
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]



### worker/worker.py — containerized task worker (image used by orchestrator)

# worker/worker.py
import os, json, time, sys

def main():
    payload_json = os.environ.get("TASK_PAYLOAD", "{}")
    try:
        payload = json.loads(payload_json)
    except Exception:
        payload = {"error": "invalid_payload"}
    # Simulate work:
    kind = payload.get("kind", "noop")
    time.sleep(1)
    if kind == "compute":
        x = payload.get("x", 1)
        y = payload.get("y", 2)
        out = {"result": x + y}
    elif kind == "echo":
        out = {"echo": payload.get("text", "")}
    else:
        out = {"note": f"no-op for {kind}"}
    # write to stdout as JSON - orchestrator captures container logs
    print(json.dumps({"status":"ok","result": out}))
    sys.exit(0)

if __name__ == "__main__":
    main()


## worker/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY worker.py /app/worker.py
CMD ["python", "worker.py"]



### orchestrator/models.py — pydantic models for validation

# orchestrator/models.py
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ToolResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    next_tasks: Optional[List[Dict[str, Any]]] = []

class AgentResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = {}
    memories: Optional[List[Dict[str, Any]]] = []
    next_tasks: Optional[List[Dict[str, Any]]] = []







### orchestrator/agents.py — agent implementations


# orchestrator/agents.py
import os
import httpx
import asyncio
import json
import docker  # docker-py
from typing import Dict, Any, List, Optional
from .models import AgentResponse, ToolResponse
from pydantic import ValidationError

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class BaseAgent:
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self._skills = {}

    def register_skill(self, name: str, func):
        self._skills[name] = func

    async def handle(self, action: str, payload: Dict[str, Any], ctx: Dict[str, Any]):
        func = self._skills.get(action)
        if not func:
            return {"status":"error","error":f"missing_skill:{action}"}
        try:
            maybe = func(payload, ctx)
            if asyncio.iscoroutine(maybe):
                out = await maybe
            else:
                out = maybe
            # validate top-level agent response
            try:
                AgentResponse.parse_obj(out)
            except ValidationError as ve:
                return {"status":"error","error":"invalid_agent_output", "details": str(ve)}
            return out
        except Exception as e:
            return {"status":"error","error": str(e)}

# ToolAgent: calls MCP tool server (HTTP)
class ToolAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, mcp_base: str):
        super().__init__(agent_id, name)
        self.mcp_base = mcp_base.rstrip("/")

    async def call_tool(self, tool_name: str, args: Dict[str, Any]):
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(f"{self.mcp_base}/call/{tool_name}", json={"args": args})
            if r.status_code >= 400:
                raise RuntimeError(f"tool_call_error:{r.status_code}:{r.text}")
            obj = r.json()
            # validate tool response shape
            ToolResponse.parse_obj(obj)
            return obj

    def register_wrapped(self, tool_name: str):
        # create a skill that calls the tool
        async def skill(payload, ctx):
            args = payload.get("args", payload.get("metadata", {}))
            tool_res = await self.call_tool(tool_name, args)
            return {"status":"ok", "result": tool_res.get("result", {}), "memories": [{"type":"tool", "tool":tool_name, "output": tool_res.get("result", {})}], "next_tasks": tool_res.get("next_tasks", [])}
        self.register_skill(f"tool_{tool_name}", skill)

# OpenAI agent (calls gpt4o)
class OpenAIAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, api_key: str):
        super().__init__(agent_id, name)
        self.api_key = api_key

    async def call_gpt4o(self, prompt: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o",
            "messages": [{"role":"user", "content": prompt}],
            "max_tokens": 512
        }
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    def register_default(self):
        # register an action that asks GPT4o to plan/subtask
        async def ask_gpt(payload, ctx):
            prompt = payload.get("prompt", "")
            resp = await self.call_gpt4o(prompt)
            # parse text content (model returns choices -> message -> content)
            text = ""
            try:
                choices = resp.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    text = msg.get("content") or choices[0].get("text", "")
            except Exception:
                text = str(resp)
            # Simple heuristic: if text contains "run:" lines, create subtasks
            next_tasks = []
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for i, line in enumerate(lines):
                # if line looks like "TOOL: run_shell echo hello" -> create a next_task
                if line.lower().startswith("tool:"):
                    # format: TOOL: toolname argkey=argval ...
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        toolname = parts[1].strip()
                        arg_text = parts[2] if len(parts) >= 3 else ""
                        # parse simple k=v pairs
                        args = {}
                        for kv in arg_text.split():
                            if "=" in kv:
                                k,v = kv.split("=",1)
                                args[k]=v
                        next_tasks.append({"name": f"auto_{toolname}_{i}", "agent_id": "tool_agent", "action": f"tool_{toolname}", "inputs": [], "metadata": {"args": args}})
            return {"status":"ok", "result":{"text": text}, "memories":[{"type":"llm","model":"gpt4o","content":text}], "next_tasks": next_tasks}
        self.register_skill("ask_gpt4o", ask_gpt)

# Reviewer: policy checks to avoid unsafe commands
import re
class ReviewerAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, blocked_patterns: Optional[List[str]] = None):
        super().__init__(agent_id, name)
        self.blocked_patterns = blocked_patterns or [r"rm\s+-rf", r"sudo", r"shutdown", r"reboot", r">", r"\|\|", r";"]
    async def check_plan(self, payload, ctx):
        plan = payload.get("plan", [])
        for step in plan:
            # if a run_shell or tool_run_shell present, check metadata.args.cmd or metadata.cmd
            if step.get("action", "").endswith("run_shell") or step.get("action","").endswith("tool_run_shell"):
                cmd = ""
                md = step.get("metadata", {}) or {}
                if "args" in md and isinstance(md["args"], dict):
                    cmd = md["args"].get("cmd","")
                cmd = cmd or md.get("cmd","")
                for pat in self.blocked_patterns:
                    if re.search(pat, cmd):
                        return {"status":"error", "decision":"reject", "reason": f"blocked pattern {pat} in cmd"}
        return {"status":"ok", "decision":"accept"}

    def register_default(self):
        self.register_skill("policy_check", self.check_plan)

# Executor that spawns Docker containers for tasks
class ContainerExecutorAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, worker_image: str = "task_worker:latest", docker_url: str = "unix://var/run/docker.sock"):
        super().__init__(agent_id, name)
        # docker client
        self.client = docker.from_env()
        self.worker_image = worker_image

    async def run_container_task(self, payload, ctx):
        """
        payload.metadata.args or payload.args contains container run arguments (e.g., TASK_PAYLOAD)
        Orchestrator will capture logs and return result.
        """
        args = payload.get("metadata", {}).get("args") or payload.get("args") or {}
        # pass TASK_PAYLOAD via env
        env = {"TASK_PAYLOAD": json.dumps(args)}
        # run container (detached) and wait for completion (simple)
        try:
            container = self.client.containers.run(self.worker_image, environment=env, detach=True, remove=True)
            # wait with timeout
            result = container.wait(timeout=30)
            logs = container.logs(stdout=True, stderr=True).decode()
            return {"status":"ok", "result": {"stdout": logs}, "memories": [{"type":"exec","output": logs}]}
        except Exception as e:
            return {"status":"error", "error": str(e)}
    def register_default(self):
        self.register_skill("run_container", self.run_container_task)










### orchestrator/engine.py — orchestration engine, graph, discovery, memory


# orchestrator/engine.py
import asyncio
import sqlite3
import os
from typing import Dict, Any, List
from .agents import ToolAgent, OpenAIAgent, ReviewerAgent, ContainerExecutorAgent, BaseAgent
from .models import ToolResponse, AgentResponse
from pydantic import ValidationError

class MemoryManager:
    def __init__(self, path: str = "/data/orch_memory.db"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT, ts REAL, item TEXT)""")
        self._short = {}  # session_id -> list
    def add_short(self, session_id: str, item: Dict[str,Any]):
        self._short.setdefault(session_id, []).append(item)
    def list_short(self, session_id: str):
        return list(self._short.get(session_id, []))
    def add_long(self, user_id: str, item: Dict[str,Any]):
        import time, uuid
        mem_id = "mem_" + uuid.uuid4().hex[:8]
        self.conn.execute("INSERT INTO memories (id,user_id,ts,item) VALUES (?,?,?,?)", (mem_id, user_id, time.time(), json.dumps(item)))
        self.conn.commit()
    def recall_long(self, user_id: str, limit: int = 20):
        cur = self.conn.cursor()
        cur.execute("SELECT item FROM memories WHERE user_id = ? ORDER BY ts DESC LIMIT ?", (user_id, limit))
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

# Simple TaskNode & Graph
class TaskNode:
    def __init__(self, node_id, name, agent_id, action, inputs=None, metadata=None):
        self.node_id = node_id; self.name = name; self.agent_id = agent_id; self.action = action
        self.inputs = inputs or []; self.metadata = metadata or {}
        self.status = "pending"; self.attempts = 0; self.result = None

    def to_dict(self):
        return {"node_id": self.node_id, "name": self.name, "agent_id": self.agent_id, "action": self.action, "inputs": self.inputs, "metadata": self.metadata, "status": self.status, "result": self.result}

class Orchestrator:
    def __init__(self, mcp_base: str, openai_key: str, worker_image: str = "task_worker:latest"):
        self.mcp_base = mcp_base
        self.memory = MemoryManager()
        # agent registry
        self.agents: Dict[str, BaseAgent] = {}
        # create tool agent & discover tools later
        self.tool_agent = ToolAgent("tool_agent", "ToolAgent", mcp_base)
        self.register_agent(self.tool_agent)
        # openai agent
        self.openai_agent = OpenAIAgent("openai_agent", "OpenAIAgent", openai_key)
        self.openai_agent.register_default()
        self.register_agent(self.openai_agent)
        # reviewer
        self.reviewer = ReviewerAgent("reviewer", "ReviewerAgent")
        self.reviewer.register_default()
        self.register_agent(self.reviewer)
        # container executor
        self.executor = ContainerExecutorAgent("executor", "ContainerExecutor", worker_image=worker_image)
        self.executor.register_default()
        self.register_agent(self.executor)
        # graph storage
        self.nodes: Dict[str, TaskNode] = {}
        self._parents = {}
        self._children = {}
        self._sem = asyncio.Semaphore(4)

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent

    async def discover_tools_and_wrap(self):
        # query MCP server to list tools and register tool_agent skills
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.mcp_base}/.well-known/tools")
            r.raise_for_status()
            tools = r.json().get("tools", [])
        for t in tools:
            name = t.get("name")
            self.tool_agent.register_wrapped(name)

    def add_node(self, node: TaskNode):
        if node.node_id in self.nodes:
            raise ValueError("node_already_exists")
        self.nodes[node.node_id] = node
        for p in node.inputs:
            self._children.setdefault(p, []).append(node.node_id)
            self._parents.setdefault(node.node_id, []).append(p)

    def _ready_nodes(self):
        ready = []
        for n in self.nodes.values():
            if n.status != "pending": continue
            parents = self._parents.get(n.node_id, [])
            if all(self.nodes[p].status == "done" for p in parents):
                ready.append(n)
        return ready

    def _payload_for(self, node: TaskNode):
        inputs = {p: self.nodes[p].result for p in node.inputs}
        return {"action": node.action, "inputs": inputs, "metadata": node.metadata}

    async def _execute_node(self, node: TaskNode, session: Dict[str,Any]):
        async with self._sem:
            node.status = "running"; node.attempts += 1
            agent = self.agents.get(node.agent_id)
            if not agent:
                node.status = "error"; node.result = {"status":"error","error":"missing_agent"}; return
            payload = self._payload_for(node)
            resp = await agent.handle(node.action, payload, {"session": session, "memory": self.memory})
            node.result = resp
            node.status = "done" if resp.get("status")=="ok" else "error"
            # store memories
            for m in resp.get("memories", []):
                self.memory.add_short(session["session_id"], m)
                self.memory.add_long(session["user_id"], m)
            # add dynamic next_tasks
            for nt in resp.get("next_tasks", []):
                nid = nt.get("node_id") or ("dyn_" + str(len(self.nodes)+1))
                new_node = TaskNode(node_id=nid, name=nt.get("name",nid), agent_id=nt.get("agent_id"), action=nt.get("action"), inputs=nt.get("inputs", []), metadata=nt.get("metadata", {}))
                try:
                    self.add_node(new_node)
                except Exception:
                    pass

    async def execute_graph(self, session: Dict[str,Any], max_iters: int = 1000):
        iterations = 0
        tasks: List[asyncio.Task] = []
        while True:
            iterations += 1
            if iterations > max_iters:
                raise RuntimeError("max_iters")
            ready = self._ready_nodes()
            if not ready:
                break
            tasks = [asyncio.create_task(self._execute_node(n, session)) for n in ready]
            if tasks:
                await asyncio.gather(*tasks)
        return {"status":"ok", "report": {nid: n.to_dict() for nid,n in self.nodes.items()}}

    # high-level helper: run from a user prompt
    async def run_from_prompt(self, user_id: str, session_id: str, prompt: str):
        # discover tools
        await self.discover_tools_and_wrap()
        # 1) create a planner node (ask openai to plan)
        plan_node = TaskNode(node_id="plan_1", name="PlanStep", agent_id="openai_agent", action="ask_gpt4o", 
                             
                             metadata={"prompt": f"Given the user request: {prompt}\nReturn a short plan. 
                                       If you want the orchestrator to call an external tool, include lines 
                                       like: TOOL: fake_search query=<term> or TOOL: run_shell cmd=<cmd>."})
        self.add_node(plan_node)
        # 2) add an aggregator node that will summarise results at the end (created up front or dynamically)
        agg_node = TaskNode(node_id="agg_1", name="Aggregate", agent_id="openai_agent", action="ask_gpt4o", inputs=["plan_1"], metadata={"prompt":"Summarise outputs: {inputs}"})
        self.add_node(agg_node)
        session = {"session_id": session_id, "user_id": user_id}
        report = await self.execute_graph(session)
        return report







## orchestrator/app.py — FastAPI entrypoint


# orchestrator/app.py
from fastapi import FastAPI, HTTPException
import os
from .engine import Orchestrator
import uvicorn

app = FastAPI(title="Orchestrator API")

MCP_BASE = os.environ.get("MCP_TOOL_URL", "http://mcp_tool:9000")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

orch = Orchestrator(mcp_base=MCP_BASE, openai_key=OPENAI_KEY, worker_image="task_worker:latest")

@app.on_event("startup")
async def startup():
    # ensure tools discovered at startup (optional)
    await orch.discover_tools_and_wrap()

@app.post("/run")
async def run(payload: dict):
    user_id = payload.get("user_id", "user_demo")
    session_id = payload.get("session_id", "sess_demo")
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")
    report = await orch.run_from_prompt(user_id, session_id, prompt)
    return report

# run with: uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("orchestrator.app:app", host="0.0.0.0", port=8000, reload=False)


#orchestrator/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ./orchestrator /app/orchestrator
RUN pip install fastapi "uvicorn[standard]" httpx pydantic docker sqlite3
EXPOSE 8000
CMD ["uvicorn", "orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000"]

'''
Next steps I can do for you (pick any)

Replace the mock MCP server with an official MCP server library (stdio or streamable HTTP) and show exact MCP negotiation.

Improve the LLM parsing & extraction rules (use structured JSON responses from the model rather than plaintext heuristics).

Add a web UI to visualize the dynamic DAG as it grows and executes.

Add robust job queue (Redis + RQ / Celery) and demonstrate worker autoscaling instead of direct Docker socket access.

If you want, I can now:

Drop each file into a single code block so you can copy/paste easily into files, or

Produce a git repo tarball content printed here, or

Implement one of the "Next steps" above.

Which would you like me to do next?

'''











# tools/stream_tool.py

from pydantic import BaseModel, Field
from typing import AsyncGenerator
import asyncio
import logging
import aiofiles
from mcp.server import ToolContext, ToolStreamEvent

logger = logging.getLogger(__name__)

'''
How the agent uses it
While streaming, GPT-4o can:

summarize partial results

detect errors

spawn new subtasks

update the user interface live

✔️ Rewrite stream_tool for real production use

Below is a much more realistic production-grade streaming tool using:

Pydantic validation

async generators

proper exception handling

chunked streaming using MCP’s event spec

cancellation support

logging

real long-running work
'''


class StreamInput(BaseModel):
    action: str = Field(..., description="The action to perform: 'read_file', 'transcribe', etc.")
    path: str = Field(..., description="File path for streaming")
    chunk_size: int = Field(4096, description="Number of bytes per streamed chunk")


async def stream_file(path: str, chunk_size: int) -> AsyncGenerator[bytes, None]:
    #aiofiles is an asynchronous file I/O library.
    async with aiofiles.open(path, "rb") as f: #opens the file without blocking the event loop.
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(0)  # allow cancellation


async def stream_tool(ctx: ToolContext, payload: StreamInput):
    """
    A production-grade streaming tool used for reading large files, logs, datasets,
    or streaming intermediate outputs from long-running processes.
    """

    try:
        await ctx.emit(ToolStreamEvent(event="tool.stream.start", data={"path": payload.path}))

        async for chunk in stream_file(payload.path, payload.chunk_size):
            await ctx.emit(
                ToolStreamEvent(event="tool.stream.chunk",
                                data={"bytes": chunk.hex(), "len": len(chunk)})
            )

        await ctx.emit(ToolStreamEvent(event="tool.stream.end", data={"status": "completed"}))

    except asyncio.CancelledError:
        logger.warning("Stream tool cancelled")
        await ctx.emit(ToolStreamEvent(event="tool.stream.cancelled", data={}))
        raise

    except Exception as e:
        logger.exception("Stream tool failed")
        await ctx.emit(ToolStreamEvent(event="tool.stream.error", data={"error": str(e)}))
        raise






## EXAMPLE

class StreamInput(BaseModel):
    action: str = Field(
        ...,
        description="Operation to perform (e.g. 'read_file')",
        example="read_file"
    )

    path: str = Field(
        ...,
        description="Path to the file to read",
        example="/var/log/system.log"
    )

    chunk_size: int = Field(
        4096,
        description="Max size (bytes) of each streamed chunk",
        ge=1,
        le=1024*1024,  # up to 1 MB
        example=8192
    )

    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata",
        example={"request_id": "abc123"}
    )



'''
What is the alias used for in Field()?

How this metadata is used by
- LLM tools (ChatGPT, Claude, OpenAI Assistants, etc.)?
- Auto-generated UIs?
- Autocomplete systems?
- Validation layers?
- Debugging tools?



While streaming, GPT-4o can:

summarize partial results

detect errors

spawn new subtasks

update the user interface live




TODO:
1. thread vs coroutine vs multiprocess
2. starting new containers
3. back and forth from gpt4 to fill the fields (pydantic validation) and 
'''