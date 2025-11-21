
import httpx
from pydantic import BaseModel, ValidationError

class TaskOutput(BaseModel):
    status: str
    result: dict
    next_tasks: list = []

class Orchestrator:
    def __init__(self):
        self.mcp_url = "http://mcp-tool:9000"

    async def call_gpt4o(self, prompt: str):
        async with httpx.AsyncClient() as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": "Bearer YOUR_KEY"},
                json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}]}
            )
            r.raise_for_status()
            return r.json()

    async def call_mcp_tool(self, tool_name: str, args: dict):
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{self.mcp_url}/call/{tool_name}", json=args)
            r.raise_for_status()
            return r.json()

    async def run(self, payload: dict):
        prompt = payload.get("prompt")
        llm_out = await self.call_gpt4o(f"Plan steps: {prompt}")
        plan = llm_out["choices"][0]["message"]["content"]

        # simple branching
        tool_calls = []
        if "search" in plan:
            tool_calls.append(("search", {"query": prompt}))
        if "bash" in plan:
            tool_calls.append(("bash_exec", {"cmd": "echo hello"}))

        outputs = []
        for name, args in tool_calls:
            res = await self.call_mcp_tool(name, args)
            try:
                validated = TaskOutput(**res)
            except ValidationError as e:
                validated = {"status":"error","error":str(e)}
            outputs.append(validated)

        return {"plan": plan, "tool_outputs": outputs}
