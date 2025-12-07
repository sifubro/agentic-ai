
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END

class State(dict):
    tasks: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    messages: List[str]




def planner(state: State):
    user_query = state["messages"][-1]

    result = llm.plan(user_query, state["results"])  # pseudo-code

    if result["action"] == "tool":
        state["tasks"].append({
            "type": "tool",
            "tool_name": result["tool_name"],
            "args": result["args"]
        })
        return {"next": "executor"}

    if result["action"] == "subtask":
        for t in result["subtasks"]:
            state["tasks"].append({"type": "subtask", "query": t})
        return {"next": "executor"}

    if result["action"] == "respond":
        return {"final_answer": result["answer"]}

    return {"next": "planner"}



def executor(state: State):
    if not state["tasks"]:
        return {"next": "summarizer"}

    task = state["tasks"].pop(0)

    if task["type"] == "tool":
        output = mcp_tools[task["tool_name"]](**task["args"])
    else:
        # sub-agent LLM
        output = llm_subagent(task["query"])

    state["results"].append({"task": task, "output": output})
    return {"next": "planner"}





graph = StateGraph(State)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("summarizer", summarizer)

graph.set_entry_point("planner")



graph.add_conditional_edges(
    "planner",  #The source node (where the conditional check happens)
    #This function receives the state after planner ran.
    lambda s: s.get("next", "planner"),  #A routing function (reads the state and returns a “label”)
    {
        "executor": "executor",  #A mapping from label → destination node
        "planner": "planner"
    }
)

graph.add_conditional_edges(
    "executor",
    lambda s: s.get("next", "planner"),
    {
        "planner": "planner",
        "summarizer": "summarizer"
    }
)

graph.set_finish_point("summarizer")

app = graph.compile()