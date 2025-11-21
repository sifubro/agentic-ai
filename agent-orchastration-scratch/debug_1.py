
def _build_payload(self, node: TaskNode) -> Dict[str, Any]:
    payload_inputs = {}
    for p in node.inputs:
        payload_inputs[p] = self._nodes[p].result
    return {"action": node.action, "inputs": payload_inputs, "metadata": node.metadata}



load_node = TaskNode(
    node_id="load_node",
    name="Load Article",
    agent_id="agent_loader",
    action="load_article",
    inputs=[],
    metadata={"url": "https://example.com"}
)
# result = {"status": "ok", "content": "Loaded article from https://example.com"}

'''
payload
{
    "action": "summarize",
    "inputs": {
        "load_node": {
            "status": "ok",
            "content": "Loaded article from https://example.com"
        }
    },
    "metadata": {"length": 50}
}
'''

summarize_node = TaskNode(
    node_id="summarize_node",
    name="Summarize Article",
    agent_id="agent_summarizer",
    action="summarize",
    inputs=["load_node"],  # <-- depends on load_node
    metadata={"length": 50}
)


def load_article(payload, ctx):
    url = payload["metadata"]["url"]
    return {"status": "ok", "content": f"Loaded article from {url}"}

def summarize(payload, ctx):
    # The result of 'load_node' is automatically here:
    article = payload["inputs"]["load_node"]["content"]
    return {"status": "ok", "summary": article[:20] + "..."}



for nt in resp.get("next_tasks", []):
    nid = nt.get("node_id") or generate_id("dyn")
    new_node = TaskNode(
        node_id=nid,
        name=nt.get("name", nid),
        agent_id=nt.get("agent_id"),
        action=nt.get("action"),
        inputs=nt.get("inputs", []),
        metadata=nt.get("metadata", {})
    )
    self.add_node(new_node)


    
#########################################

planner = Agent("planner", "Planner")
researcher = Agent("researcher", "Researcher")
summarizer = Agent("summarizer", "Summarizer")


def planning_skill(payload, ctx):
    topic = payload["metadata"]["topic"]
    return {
        "status": "ok",
        "message": f"Planning research on {topic}",
        "next_tasks": [
            {
                "name": "Research Topic",
                "agent_id": "researcher",
                "action": "research",
                "metadata": {"topic": topic}
            },
            {
                "name": "Summarize Findings",
                "agent_id": "summarizer",
                "action": "summarize",
                "inputs": ["Research Topic"]
            }
        ]
    }


def research_skill(payload, ctx):
    topic = payload["metadata"]["topic"]
    return {"status": "ok", "findings": f"Collected data on {topic}"}

def summarize_skill(payload, ctx):
    findings = payload["inputs"]["Research Topic"]["findings"]
    return {"status": "ok", "summary": findings[:30] + "..."}



planner.register_skill("plan", planning_skill)
researcher.register_skill("research", research_skill)
summarizer.register_skill("summarize", summarize_skill)


plan_node = TaskNode(
    node_id="planner_node",
    name="Planner Task",
    agent_id="planner",
    action="plan",
    inputs=[],
    metadata={"topic": "climate change"}
)
orch.add_node(plan_node)



await orch.execute_graph("sess123")

##############################


def plan_skill(payload, ctx):
    question = ctx["session"]["question"]
    print(f"[Planner] Planning for question: {question}")
    return {
        "status": "ok",
        "plan": ["Research", "Analyze", "Summarize"],
        "next_tasks": [
            {
                "name": "Research Topic",
                "agent_id": "researcher",
                "action": "research",
                "metadata": {"topic": question}
            },
            {
                "name": "Analyze Research",
                "agent_id": "analyzer",
                "action": "analyze",
                "inputs": ["Research Topic"]
            },
            {
                "name": "Summarize Analysis",
                "agent_id": "summarizer",
                "action": "summarize",
                "inputs": ["Analyze Research"]
            }
        ]
    }


def research_skill(payload, ctx):
    topic = payload["metadata"]["topic"]
    print(f"[Researcher] Searching data for topic: {topic}")
    # Simulate multiple sources
    data = [f"Fact 1 about {topic}", f"Fact 2 about {topic}", f"Fact 3 about {topic}"]
    ctx["memory"].add_short(ctx["session"]["session_id"], {"topic": topic, "facts": data})
    return {"status": "ok", "findings": data}


def analyze_skill(payload, ctx):
    findings = payload["inputs"]["Research Topic"]["findings"]
    print(f"[Analyzer] Processing findings: {findings}")
    analysis = f"Key themes: {', '.join([f.split()[1] for f in findings])}"
    ctx["memory"].add_short(ctx["session"]["session_id"], {"analysis": analysis})
    # Dynamically add a validation step
    return {
        "status": "ok",
        "analysis": analysis,
        "next_tasks": [
            {
                "name": "Validation",
                "agent_id": "validator",
                "action": "validate",
                "inputs": ["Analyze Research"]
            }
        ]
    }



def summarize_skill(payload, ctx):
    analysis = payload["inputs"]["Analyze Research"]["analysis"]
    print(f"[Summarizer] Creating summary for: {analysis}")
    summary = f"Summary: {analysis.lower()}."
    ctx["memory"].add_long(ctx["session"]["user_id"], {"summary": summary})
    return {"status": "ok", "summary": summary}



def validate_skill(payload, ctx):
    analysis = payload["inputs"]["Analyze Research"]["analysis"]
    print(f"[Validator] Validating analysis: {analysis}")
    return {"status": "ok", "validated": True}



session = {
    "session_id": "sess_001",
    "user_id": "user_123",
    "question": "What are the effects of climate change on polar bears?"
}

memory = MemoryManager()
Short-term: {}
Long-term: {}


## Step 1: Initial Graph Creation
planner_node = TaskNode(
    node_id="plan_node",
    name="Plan Research Workflow",
    agent_id="planner",
    action="plan",
    inputs=[],
    metadata={}
)
orch.add_node(planner_node)


'''

▶️ Step 2: Execute Graph (Iteration 1)
READY NODES:
plan_node (no dependencies)

MESSAGE sent to planner agent:
{
  "from": "sess_001",
  "to": "planner",
  "task_id": "plan_node",
  "payload": {
    "action": "plan",
    "inputs": {},
    "metadata": {}
  },
  "metadata": {"node_name": "Plan Research Workflow", "action": "plan"}
}

CONTEXT:
{
  "agent": {"id": "planner", "role": "assistant"},
  "session": {"session_id": "sess_001", "user_id": "user_123", "question": "What are the effects of climate change on polar bears?"},
  "memory": memory
}

RETURNS:
{
  "status": "ok",
  "next_tasks": [
    {"name": "Research Topic", "agent_id": "researcher", "action": "research", "metadata": {"topic": "What are the effects of climate change on polar bears?"}},
    {"name": "Analyze Research", "agent_id": "analyzer", "action": "analyze", "inputs": ["Research Topic"]},
    {"name": "Summarize Analysis", "agent_id": "summarizer", "action": "summarize", "inputs": ["Analyze Research"]}
  ]
}




▶️ Step 3: Execute Graph (Iteration 2)
READY NODES:
Research Topic (no parents)

MESSAGE sent to researcher:
{
  "action": "research",
  "inputs": {},
  "metadata": {"topic": "What are the effects of climate change on polar bears?"}
}

OUTPUT:
{
  "status": "ok",
  "findings": [
    "Fact 1 about What are the effects of climate change on polar bears?",
    "Fact 2 about What are the effects of climate change on polar bears?",
    "Fact 3 about What are the effects of climate change on polar bears?"
  ]
}

✅ Short-term memory is updated:
{
  "sess_001": {
    "topic": "What are the effects of climate change on polar bears?",
    "facts": [...]
  }
}


▶️ Step 4: Execute Graph (Iteration 3)
READY NODES:
Analyze Research (its parent Research Topic is done)

Payload built:
{
  "action": "analyze",
  "inputs": {
    "Research Topic": {
      "findings": ["Fact 1 ...", "Fact 2 ...", "Fact 3 ..."]
    }
  },
  "metadata": {}
}

OUTPUT:
{
  "status": "ok",
  "analysis": "Key themes: 1, 2, 3",
  "next_tasks": [
    {"name": "Validation", "agent_id": "validator", "action": "validate", "inputs": ["Analyze Research"]}
  ]
}

✅ Short-term memory updated:
{
  "sess_001": {
    "analysis": "Key themes: 1, 2, 3"
  }
}


▶️ Step 5: Execute Graph (Iteration 4)
READY NODES:
- Summarize Analysis (depends on Analyze Research)
- Validation (depends on Analyze Research)
The orchestrator can run both concurrently since their dependencies are satisfied.


Summarizer payload:
{
  "inputs": {
    "Analyze Research": {"analysis": "Key themes: 1, 2, 3"}
  },
  "metadata": {}
}

Validator payload:
{
  "inputs": {
    "Analyze Research": {"analysis": "Key themes: 1, 2, 3"}
  },
  "metadata": {}
}


Summarizer executes:
[Summarizer] Creating summary for: Key themes: 1, 2, 3
Output:
{"status": "ok", "summary": "Summary: key themes: 1, 2, 3."}

✅ Long-term memory:
{
  "user_123": {
    "summary": "Summary: key themes: 1, 2, 3."
  }
}

Validator executes:
[Validator] Validating analysis: Key themes: 1, 2, 3
OUTPUT
{"status": "ok", "validated": true}


🧩 Step 6: Graph Completion
✅ Final memory state:
Short-term: {
  "sess_001": {
    "topic": "What are the effects of climate change on polar bears?",
    "facts": ["Fact 1 ...", "Fact 2 ...", "Fact 3 ..."],
    "analysis": "Key themes: 1, 2, 3"
  }
}

Long-term: {
  "user_123": {
    "summary": "Summary: key themes: 1, 2, 3."
  }
}


✅ Final report from orchestrator:
{
  "status": "ok",
  "report": {
    "plan_node": {"status": "done"},
    "Research Topic": {"status": "done", "result": {"findings": [...]}},
    "Analyze Research": {"status": "done", "result": {"analysis": "Key themes: 1, 2, 3"}},
    "Validation": {"status": "done", "result": {"validated": true}},
    "Summarize Analysis": {"status": "done", "result": {"summary": "Summary: key themes: 1, 2, 3."}}
  }
}


'''



























