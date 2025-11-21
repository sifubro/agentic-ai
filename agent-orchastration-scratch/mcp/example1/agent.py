# inside your agent system
from mcp.client import MCPClient  # hypothetical import
client = await MCPClient.create_stdio(command="python", args=["mcp_calc_server.py"])

tools = await client.list_tools()   # e.g., [{"name":"add", "schema":...}, {"name":"multiply",...}]

agent.register_skill("calc_add", lambda payload, ctx: client.call_tool("add", payload))
agent.register_skill("calc_multiply", lambda payload, ctx: client.call_tool("multiply", payload))


node = TaskNode(
    node_id="calc1",
    name="AddNumbers",
    agent_id="calc_agent",
    action="calc_add",
    inputs=[],
    metadata={"a": 15, "b": 27}
)

'''
Agent handles the message, sees action calc_add, calls the add tool via MCP with {a:15, b:27}, returns result (42).

Memory can store {"tool_used":"add","args":{"a":15,"b":27},"result":42}.

'''

