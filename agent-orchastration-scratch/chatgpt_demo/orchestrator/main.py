
# Simplified orchestrator HTTP API + multi-agent driver
from fastapi import FastAPI
from orchestrator.engine import Orchestrator

app = FastAPI()
orch = Orchestrator()

@app.post("/run")
async def run_task(payload: dict):
    return await orch.run(payload)
