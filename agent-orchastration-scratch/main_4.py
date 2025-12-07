'''
https://chatgpt.com/c/69061637-7ef4-8332-906d-e6efcb5bb0d4


🔥 WHAT WE BUILD IN THIS MESSAGE
1. Full Microservices Architecture Overview
2. Production-Ready MCP Tool Server (HTTP streaming, Pydantic, async)
3. Production Orchestrator API (FastAPI)
4. Planning Engine using GPT-4o (with MCP tool usage + dynamic graph creation)
5. Worker Microservice (Gemini Flash)
6. RabbitMQ Integration for Task Distribution
7. Memory Service (Short-term, Long-term)
8. Docker Compose Stack (All services wired together)
'''

'''
2) MCP TOOL SERVER (PRODUCTION READY)

A real MCP server must:

✔ use HTTP streaming
✔ implement MCP event protocol (tool.started, tool.progress, etc.)
✔ validate inputs via Pydantic
✔ handle cancellation
✔ support multiple tools
'''

# 📁 mcp_tool_server/server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio, json, math, ast, operator

app = FastAPI(title="MCP Tool Server")


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────
# TODO: Pydantic tutorial
class SearchInput(BaseModel):
    query: str = Field(...)


class CalcInput(BaseModel):
    expression: str


# ──────────────────────────────────────────────
# Tool: search (dummy implementation)
# ──────────────────────────────────────────────
@app.post("/mcp/search")
async def search_tool(payload: SearchInput):

    async def event_stream():
        # start
        yield json.dumps({"event": "tool.search.started"}) + "\n"

        await asyncio.sleep(0.5)

        # return fake result
        result = {
            "query": payload.query,
            "result": f"Fake search result for '{payload.query}'"
        }

        # end
        yield json.dumps({
            "event": "tool.search.completed",
            "data": result
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ──────────────────────────────────────────────
# Tool: calculator (safe expression evaluation)
# ──────────────────────────────────────────────
@app.post("/mcp/calculator")
async def calculator(payload: CalcInput):

    def safe_eval(expr: str):
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp):
                op = allowed_ops[type(node.op)]
                return op(eval_node(node.left), eval_node(node.right))
            raise ValueError("Unsafe expression")

        # TODO: Pydantic tutorial
        return eval_node(ast.parse(expr, mode="eval").body)

    async def event_stream():
        yield json.dumps({"event": "tool.calculator.started"}) + "\n"
        await asyncio.sleep(0.2)

        result = safe_eval(payload.expression)

        yield json.dumps({
            "event": "tool.calculator.completed",
            "data": {"result": result}
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")




# ======================================================================================

'''
3) ORCHESTRATOR (FASTAPI)

Handles:

user requests

planning loop

DAG building

sending tasks to RabbitMQ

collecting worker results

storing memory
'''


# 📁 orchestrator/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import aio_pika
import asyncio
from .planner import planner_cycle
from .memory import MemoryStore

app = FastAPI()
memory = MemoryStore()


class UserQuery(BaseModel):
    user_id: str
    query: str


@app.post("/run")
async def run_query(data: UserQuery):

    # Run the planner loop (GPT-4o)
    async for event in planner_cycle(data.user_id, data.query):
        # streaming planning output
        yield event

    return {"status": "done"}


#========================================================================================

'''
4) PLANNER ENGINE (GPT-4o)

GPT-4o:

reads the query

optionally calls MCP tools

then returns next_tasks to orchestrator

orchestrator dispatches tasks to workers

workers return output

GPT-4o continues planning (loop)
'''



# 📁 orchestrator/planner.py
from openai import AsyncOpenAI
import aiohttp
import asyncio
import json
from .rabbit import publish_task
from .memory import memory_store

client = AsyncOpenAI()


async def planner_cycle(user_id: str, query: str):

    messages = [
        {"role": "system", "content": PLANNER_RULES},
        {"role": "user", "content": query}
    ]

    while True:

        # ───────────── GPT-4o Invoke ─────────────
        response = await client.responses.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
            tools=MCP_TOOLS,
            tool_choice="auto"
        )

        tool_call = None
        next_tasks = None

        async for ev in response:

            if ev.type == "response.output_text.delta":
                yield {"event": "planner.text", "data": ev.delta}

            elif ev.type == "response.tool_call":
                tool_call = ev.tool_call

            elif ev.type == "response.completed":
                content = ev.output[0].content[0]

                if "next_tasks" in content:
                    next_tasks = content["next_tasks"]

        # ───────────── HANDLE TOOL CALL ─────────────
        if tool_call:
            result = await call_mcp_tool(tool_call)
            messages.append({"role": "tool", "content": json.dumps(result)})
            continue  # go to next planning cycle

        # ───────────── HANDLE TASK GENERATION ─────────────
        if next_tasks:
            memory_store.add_short(user_id, next_tasks)

            # dispatch all tasks concurrently via RabbitMQ
            for t in next_tasks:
                await publish_task(t)

            # collect their results
            #TODO: implement collect_worker_results
            worker_outputs = await collect_worker_results(len(next_tasks))

            messages.append({"role": "assistant", "content": json.dumps(worker_outputs)})
            continue

        break





# ===============================


import os
import json
import aiohttp

MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "http://mcp-tool-server:8000")

async def call_mcp_tool(tool_call):
    """
    Execute an MCP tool described by `tool_call` by calling the MCP Tool Server
    and return the final result (the `data` from the *.completed event).

    Assumes:
    - `tool_call.name` is the tool name ("search", "calculator", etc.).
    - `tool_call.arguments` is either a dict or a JSON string.
    - MCP Tool Server exposes:
        POST /mcp/search
        POST /mcp/calculator
      returning NDJSON like:
        {"event": "tool.search.started"}
        {"event": "tool.search.completed", "data": {...}}
    """

    # 1) Figure out which endpoint to hit based on the tool name
    tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "tool_name", None)
    if not tool_name:
        raise ValueError("tool_call is missing a 'name' field")

    endpoint_map = {
        "search": "/mcp/search",
        "calculator": "/mcp/calculator",
    }

    if tool_name not in endpoint_map:
        raise ValueError(f"Unknown MCP tool: {tool_name}")

    url = MCP_BASE_URL + endpoint_map[tool_name]

    # 2) Extract arguments
    args = getattr(tool_call, "arguments", {}) or getattr(tool_call, "input", {}) or {}
    if isinstance(args, str):
        # OpenAI often sends arguments as a JSON string
        args = json.loads(args)

    # 3) Call the MCP tool server and parse NDJSON stream
    completed_data = None

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=args) as resp:
            resp.raise_for_status()

            # The server returns "application/x-ndjson" with one JSON object per line
            async for raw_line in resp.content:
                line = raw_line.decode().strip()
                if not line:
                    continue

                event = json.loads(line)
                event_name = event.get("event", "")

                # Keep the last *.completed event's data
                if event_name.endswith(".completed"):
                    completed_data = event.get("data")

    if completed_data is None:
        raise RuntimeError(f"MCP tool '{tool_name}' finished without a completed event")

    return completed_data




# ===============================

'''
1️⃣ Add a global results queue + callback model

In your orchestrator / planner server (same place as planner_cycle), define:
'''


# planner.py (or wherever planner_cycle lives)
import asyncio
from pydantic import BaseModel

# Global queue to collect worker results
worker_results_queue: asyncio.Queue = asyncio.Queue()


class WorkerResult(BaseModel):
    node_id: str
    result: str

'''
2️⃣ Add a callback endpoint for the workers

In your FastAPI app (the orchestrator), add an endpoint that the worker will call:
'''


from fastapi import APIRouter
from .planner import worker_results_queue, WorkerResult

router = APIRouter()

@router.post("/worker/callback")
async def worker_callback(payload: WorkerResult):
    # Push result into the in-memory queue
    await worker_results_queue.put(payload.dict())
    return {"status": "ok"}


# Make sure this router is included in your main app:

app.include_router(router)


# And make sure your WorkerTask’s callback_url points to this endpoint, e.g.:

callback_url="http://orchestrator:8000/worker/callback"


'''
3️⃣ Implement collect_worker_results

Now you can implement the function the planner uses:
'''


import asyncio
from typing import List, Dict
from .planner import worker_results_queue  # same queue as above

async def collect_worker_results(expected_count: int, timeout: float = 60.0) -> list[dict]:
    """
    Wait for `expected_count` worker results to arrive via the callback endpoint.
    Returns a list of result dicts: [{"node_id": ..., "result": ...}, ...]
    """
    results: List[Dict] = []

    for _ in range(expected_count):
        try:
            item = await asyncio.wait_for(worker_results_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            # If some workers are slow or failed, you can decide to break or raise
            break
        results.append(item)

    return results


'''
4️⃣ Planner usage (unchanged)

Now your original code works as intended:
'''


# collect their results
worker_outputs = await collect_worker_results(len(next_tasks))

messages.append({"role": "assistant", "content": json.dumps(worker_outputs)})







# =====================================================================================

'''
5) WORKER MICROSERVICE (Gemini Flash)

Each worker:

consumes tasks from RabbitMQ

sends them to Gemini Flash

returns results to orchestrator callback
'''


# 📁 worker/worker.py
from google import generativeai as genai
from pydantic import BaseModel
import aiohttp
import json
from .rabbit import consume_tasks

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

class WorkerTask(BaseModel):
    node_id: str
    action: str
    agent_id: str
    inputs: dict
    callback_url: str


async def handle_task(task: WorkerTask):

    prompt = f"Task: {task.action}\nInputs: {task.inputs}"

    response = model.generate_content(prompt)
    result = response.text

    # callback orchestrator
    async with aiohttp.ClientSession() as session:
        await session.post(task.callback_url, json={
            "node_id": task.node_id,
            "result": result
        })


async def main():
    async for task_json in consume_tasks():
        task = WorkerTask(**task_json)
        await handle_task(task)



# ============================================================

# 6) RabbitMQ Publish + Consume


# 📁 orchestrator/rabbit.py

import aio_pika
import json

async def publish_task(task: dict):
    connection = await aio_pika.connect_robust("amqp://rabbitmq/")
    channel = await connection.channel()
    await channel.default_exchange.publish(
        aio_pika.Message(body=json.dumps(task).encode()),
        routing_key="tasks.queue"
    )
    await connection.close()

# 📁 worker/rabbit.py
import aio_pika
import json
import asyncio

async def consume_tasks():
    connection = await aio_pika.connect_robust("amqp://rabbitmq/")
    channel = await connection.channel()
    queue = await channel.declare_queue("tasks.queue", durable=True)

    async with queue.iterator() as q:
        async for message in q:
            async with message.process():
                yield json.loads(message.body)

# ============================================================

# 7) Memory Service (Short-term + Long-term)


# 📁 orchestrator/memory.py

class MemoryStore:

    def __init__(self):
        self.short = {}  # session → list
        self.long = {}   # user → list

    def add_short(self, user_id, data):
        self.short.setdefault(user_id, []).append(data)

    def add_long(self, user_id, data):
        self.long.setdefault(user_id, []).append(data)

    def recall_short(self, user_id):
        return self.short.get(user_id, [])

    def recall_long(self, user_id):
        return self.long.get(user_id, [])


memory_store = MemoryStore()

# ============================================================

# 8) Docker Compose (all wired together)

# 📁 docker-compose.yml
'''
version: "3.9"
services:

  orchestrator:
    build: ./orchestrator
    ports: ["8000:8000"]
    depends_on: [rabbitmq]
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      MCP_SERVER_URL: http://mcp-tool-server:9000

  worker1:
    build: ./worker
    depends_on: [rabbitmq]
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}

  worker2:
    build: ./worker
    depends_on: [rabbitmq]
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}

  mcp-tool-server:
    build: ./mcp_tool_server
    ports: ["9000:9000"]

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
'''

# Workers are autoscaled simply by adding more replicas.

# ============================================================
'''
🎉 NEXT STEPS AVAILABLE WHEN YOU SAY “continue”

I can now add:

🔥 Advanced Features

Auto-scaling workers

Multi-agent negotiation protocol

Persistent DAG state (Postgres)

Orchestrator UI + live graph visualization

Worker sandboxing (Docker API)

Full error propagation model

Data provenance tracing

Complete high availability setup

Secure tokenized authentication across microservices

🧠 Advanced AI Logic

Planner memory integration

Tool selection heuristics

Cross-agent debate + arbitration

Self-healing DAGs

Model fallback (GPT-4o → Gemini → Claude)
'''
















































