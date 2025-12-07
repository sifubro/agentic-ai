"""
LangGraph Planner Examples

Contains four runnable templates (conceptual) showing how to implement:

a) Planner using GPT-4o as the LLM (how it "plans" & expected output schema)
b) Planner wired to OpenAI's MCP tools (tool calling example)
c) Multiple sub-agents (researcher, coder, summarizer) pattern
d) DAG-style task planning (dependencies and aggregation)

Notes:
- These are templates: replace `OPENAI_API_KEY` and tool endpoints with your real values.
- The code uses pseudo wrappers for LLM calls and MCP calls; adapt to your SDK/Responses API usage.
- LangGraph API shape (StateGraph, add_node, add_conditional_edges, compile) is used.

References:
- OpenAI GPT-4o model docs: https://platform.openai.com/docs/models/gpt-4o
- OpenAI MCP / tools guides: https://platform.openai.com/docs/guides/tools-connectors-mcp and https://platform.openai.com/docs/mcp

"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import uuid
import time

# ---------------------------------------------------------------------------
# Helper LLM & MCP wrappers (replace with real OpenAI Responses API client calls)
# ---------------------------------------------------------------------------

class DummyLLM:
    """Simple wrapper to simulate LLM responses. Replace with real API calls.
    For a real deployment with GPT-4o use the Responses API client and model='gpt-4o'.
    See OpenAI docs for the Responses API and MCP tooling.
    """
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def plan(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner prompt pattern and output schema (recommended):

        - The planner MUST return a small JSON-like structure in a consistent schema so the graph
          can parse it reliably.

        Example schema produced by LLM (planner):
        {
          "plan_id": "uuid",
          "steps": [
            {"id": "task-1","type": "research","prompt": "Find citations on X","deps": []},
            {"id": "task-2","type": "tool","tool": "search_api","args": {"q": "X"},"deps": ["task-1"]}
          ],
          "final_action": "summarize" | "respond_directly",
          "explain": "Human-readable explanation"
        }

        The LLM should be instructed (via system + few-shot) to ONLY return JSON.
        """
        # In production, call OpenAI Responses API with system messages and a strong JSON-only instruction.
        # Here we return a deterministic toy plan based on the user_query for illustration.
        pid = str(uuid.uuid4())
        if "code" in user_query.lower():
            steps = [
                {"id": "t1", "type": "research", "prompt": "Find best practices for X library", "deps": []},
                {"id": "t2", "type": "coder", "prompt": "Write minimal example for X", "deps": ["t1"]},
                {"id": "t3", "type": "summarize", "prompt": "Summarize the final result", "deps": ["t2"]},
            ]
            final_action = "summarize"
        else:
            steps = [
                {"id": "t1", "type": "tool", "tool": "web_search", "args": {"q": user_query}, "deps": []},
                {"id": "t2", "type": "summarize", "prompt": "Synthesize answers", "deps": ["t1"]}
            ]
            final_action = "respond_directly"

        return {
            "plan_id": pid,
            "steps": steps,
            "final_action": final_action,
            "explain": "Auto-generated plan from GPT-4o style planner"
        }

    def call_subagent(self, role: str, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a sub-agent LLM. `role` selects behavior (researcher, coder, summarizer).
        Return structure:
          {"output": "text or structured output","meta": {...}}
        """
        # Replace with actual LLM call; here we simulate.
        return {"output": f"[{role} result for prompt: {prompt}]", "meta": {"time": time.time()}}


class MCPClient:
    """Wrapper to call external tools exposed via an MCP server. Replace with real calls.

    For OpenAI Responses API, you can expose a remote MCP server and the model will be able to
    call those tools directly. Here we simulate by calling local functions.
    """
    def __init__(self):
        self.tools = {
            "web_search": lambda q: {"hits": [f"Result for {q}"]},
            "calc": lambda expr: {"result": eval(expr)}
        }

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": "unknown tool"}
        return self.tools[tool_name](**args)

# Instantiate shared clients
llm = DummyLLM(model="gpt-4o")
mcp = MCPClient()

# ---------------------------------------------------------------------------
# Shared State shape
# ---------------------------------------------------------------------------

def make_initial_state(user_query: str) -> Dict[str, Any]:
    return {
        "user_query": user_query,
        "planner_history": [],        # records planner decisions
        "task_queue": [],             # list of tasks (dicts)
        "completed": {},              # task_id -> result
        "plan_meta": None,            # raw plan from planner
        "final_answer": None,
    }

# ---------------------------------------------------------------------------
# Version A: Planner using GPT-4o — how it plans and expected LLM output
# ---------------------------------------------------------------------------

def planner_node(state: Dict[str, Any]):
    user_q = state["user_query"]
    plan = llm.plan(user_q, {"completed": state["completed"]})

    # Validate and normalize plan (defensive)
    assert "steps" in plan and isinstance(plan["steps"], list), "planner must return steps list"

    # Put tasks into the task_queue. Each task gets a deterministic id if not present.
    for s in plan["steps"]:
        tid = s.get("id", str(uuid.uuid4()))
        s["id"] = tid
        s.setdefault("status", "pending")
        state["task_queue"].append(s)

    state["plan_meta"] = plan
    state["planner_history"].append({"plan_id": plan["plan_id"], "time": time.time()})

    # Decide next node
    if state["task_queue"]:
        return {"__next__": "executor"}
    else:
        # nothing to do: respond directly
        if plan.get("final_action") == "respond_directly":
            state["final_answer"] = plan.get("explain")
            return {"__finish__": True}
        return {"__next__": "summarizer"}

# ---------------------------------------------------------------------------
# Executor node: handles tool calls or subagents (one node executing tasks)
# ---------------------------------------------------------------------------

def executor_node(state: Dict[str, Any]):
    if not state["task_queue"]:
        return {"__next__": "planner"}

    task = state["task_queue"].pop(0)
    tid = task["id"]
    ttype = task.get("type")

    if ttype == "tool":
        result = mcp.call_tool(task.get("tool"), task.get("args", {}))
    elif ttype in ("research", "coder", "summarize", "summarizer"):
        # dispatch to a sub-agent LLM with role mapped
        role_map = {"research": "researcher", "coder": "coder", "summarize": "summarizer"}
        role = role_map.get(ttype, "researcher")
        prompt = task.get("prompt") or task.get("args", {})
        result = llm.call_subagent(role, prompt, {"completed": state["completed"]})
    else:
        # fallback: echo
        result = {"output": f"Unhandled task type: {ttype}"}

    # store completion
    state["completed"][tid] = result

    # After completing one task, go back to planner for re-evaluation
    return {"__next__": "planner"}

# ---------------------------------------------------------------------------
# Summarizer node: optional node to aggregate final results
# ---------------------------------------------------------------------------

def summarizer_node(state: Dict[str, Any]):
    # Feed the completed task outputs into a summarizer LLM (or the planner) and produce final answer
    completed = state["completed"]
    prompt = f"You are a summarizer. Consolidate these results: {completed}"
    out = llm.call_subagent("summarizer", prompt, {})
    state["final_answer"] = out["output"]
    return {"__finish__": True}

# ---------------------------------------------------------------------------
# Build the static LangGraph for the Planner pattern
# ---------------------------------------------------------------------------

def build_planner_graph():
    graph = StateGraph(dict)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.set_finish_point("summarizer")

    # Conditional routing: nodes return a dict with __next__ keys or __finish__ flag
    def planner_router(s):
        return s.get("__next__", "planner")

    def executor_router(s):
        return s.get("__next__", "planner")

    graph.add_conditional_edges("planner", lambda s: s.get("__next__", "planner"), {
        "executor": "executor",
        "summarizer": "summarizer",
        "planner": "planner",
    })

    graph.add_conditional_edges("executor", lambda s: s.get("__next__", "planner"), {
        "planner": "planner",
        "summarizer": "summarizer",
    })

    return graph

# ---------------------------------------------------------------------------
# Version B: Wiring OpenAI MCP (conceptual)
# - Here we show how the executor would call tools via MCP rather than local functions.
# - For the Responses API with MCP, the model can call the tools itself if you expose them; but
#   this example demonstrates an explicit executor that uses an MCP client wrapper.
# ---------------------------------------------------------------------------

# In this file MCPClient.call_tool is the right abstraction to replace with a real remote call.
# When using OpenAI Responses API, you may either:
#  - Register an MCP server so the LLM can call tools directly (model-driven tools)
#  - Or handle tool invocation in your executor (programmatic tools) using the same MCP descriptors.

# ---------------------------------------------------------------------------
# Version C: Multiple Sub-agents (researcher, coder, summarizer)
# - Use the same executor, but map task types to roles; optionally use different model configs.
# - Subagents can be isolated LLM calls with specialized system prompts.
# ---------------------------------------------------------------------------

# Example: specialized sub-agent wrappers (could use different models/settings)

def researcher_subagent(prompt: str, context: Dict[str, Any]):
    return llm.call_subagent("researcher", prompt, context)


def coder_subagent(prompt: str, context: Dict[str, Any]):
    # Could call a code-model variant or use more temperature tweaks
    return llm.call_subagent("coder", prompt, context)


def summarizer_subagent(prompt: str, context: Dict[str, Any]):
    return llm.call_subagent("summarizer", prompt, context)

# Replace executor logic to dispatch to those specialized wrappers as needed.

# ---------------------------------------------------------------------------
# Version D: DAG-style task planning
# - Plan.steps contains "deps" arrays and we build a small DAG scheduler inside state.
# ---------------------------------------------------------------------------

def dag_scheduler_node(state: Dict[str, Any]):
    # If no plan_meta, ask planner for one
    if not state.get("plan_meta"):
        p = llm.plan(state["user_query"], {})
        state["plan_meta"] = p

    steps = state["plan_meta"]["steps"]

    # Build helper maps
    by_id = {s["id"]: s for s in steps}
    status = state.setdefault("dag_status", {s["id"]: "pending" for s in steps})

    # Enqueue any ready tasks (deps satisfied)
    for sid, s in by_id.items():
        if status[sid] == "pending":
            deps = s.get("deps", [])
            if all(status[d] == "done" for d in deps):
                # push to queue and mark running
                state["task_queue"].append(s)
                status[sid] = "running"

    # If queue has work, go to executor
    if state["task_queue"]:
        return {"__next__": "executor"}

    # If all tasks done -> summarizer
    if all(v == "done" for v in status.values()):
        return {"__next__": "summarizer"}

    # otherwise re-evaluate (in LangGraph you'd return to planner or sleep)
    return {"__next__": "planner"}

# Modify executor_node to mark dag_status tasks as done after execution
# (left as an exercise to integrate dag_scheduler_node and executor_node in the same graph)

# ---------------------------------------------------------------------------
# How to run (example)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_q = "Please research best practices for using LangGraph with GPT-4o and write a short example code"
    state = make_initial_state(user_q)

    graph = build_planner_graph()
    app = graph.compile()

    # run the app - in LangGraph you'd pass the state in; this is illustrative
    out = app(state)
    print("Final state:", state.get("final_answer"))

'''
TODO
How do we force an LLM to return structured output?
Do not use pseudo wrappers for LLMs and MCP calls, please adapt to real time deployment scenario
Replace with actual LLM calls
Choose the instruction and docstring (add few shot examples) to Planner carefully so that is responds in JSON.
Use real output schema and structured output to force LLM to responds in JSON
OpenAI GPT-4o model docs: https://platform.openai.com/docs/models/gpt-4o
OpenAI MCP / tools guides: https://platform.openai.com/docs/guides/tools-connectors-mcp and https://platform.openai.com/docs/mcp
For MCP include MCP servers so that the MCPClient calls a real MCP server. The MCPServers should return the weather and to flight itenaries to book a flight
Do NOT use a Wrapper for MCPClient. For OpenAI Responses API, expose a remote MCP server and the model will be able to
call those tools directly. (Expose remote MCP server and call the tools via MCP - do not simulate by calling local functions)




This is my workflow right now:
- openai/o3 for planning the coding tasks and very detailed instructions
- google/2.5pro for viewing the whole code based and making adjustments + giving advise on where to start
- anthropic/4-sonnet for implementing the actual code

Are you using any coding assistants? I would recommend using Roo Code + Requesty and using 2.5 flash as an orchestrator!




I think you made a mistake. LangGraph does not create nodes dynamically at runtime (after the graph is compiled). Instead, it supports dynamic routing and conditional edges, which let a fixed set of nodes behave dynamically.

LangGraph offers “dynamic execution” using:  Dynamic DAG via “Return next step” pattern.
A node can return a function (or reference) to run next — even a generic handler.


How does the LLM decide when to call a tool or spawn subtask or just answer directly?

Please generate 

a) Version with calling gpt4o as the llm. How does it "plan" the next steps? 
How it the output of llm structured 

b) Version with OpenAI’s MCP tool calls wired in 

c) Version with multiple sub-agents (e.g. researcher, coder, summarizer) d) Version with DAG-style task planning
'''