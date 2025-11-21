client_fs = await MCPClient.create_stdio(command="python", args=["mcp_fs_server.py"])
tools_fs = await client_fs.list_tools()

fs_agent = Agent("fs_agent", "FileSystemAgent")

async def list_dir_skill(payload, ctx):
    lst = await client_fs.call_tool("list_dir", {"path": payload["path"]})
    return {"status":"ok","files":lst}

async def read_file_skill(payload, ctx):
    text = await client_fs.call_tool("read_file", {"path": payload["path"]})
    return {"status":"ok","content":text}

async def write_file_skill(payload, ctx):
    ok = await client_fs.call_tool("write_file", {"path": payload["path"], "content": payload["content"]})
    return {"status":"ok","wrote":ok}

fs_agent.register_skill("list_dir", list_dir_skill)
fs_agent.register_skill("read_file", read_file_skill)
fs_agent.register_skill("write_file", write_file_skill)



n1 = TaskNode("fs1","ListHome","fs_agent","list_dir",[],{"path":"/home/user"})
n2 = TaskNode("fs2","ReadConfig","fs_agent","read_file",["fs1"],{"path":"/home/user/config.txt"})
