
# Minimal streaming HTTP MCP tool server
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

class SearchArgs(BaseModel):
    query: str

class ExecArgs(BaseModel):
    cmd: str

@app.post("/call/search")
async def call_search(args: SearchArgs):
    # Fake search
    return {
        "status":"ok",
        "result":{"summary":f"Search results for {args.query}"},
        "next_tasks":[]
    }

@app.post("/call/bash_exec")
async def bash_exec(args: ExecArgs):
    proc = subprocess.Popen(args.cmd, shell=True, stdout=subprocess.PIPE)
    out = proc.stdout.read().decode()
    return {"status":"ok","result":{"stdout":out},"next_tasks":[]}
