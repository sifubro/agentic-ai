from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "websearch": {
        "transport": "streamable_http",
        "url": "http://localhost:9000/mcp"
    }
})
tools = await client.get_tools()

# Wrap into research agent
research_agent = Agent("researcher", "ResearchAgent")
async def do_search(payload, ctx):
    query = payload["query"]
    results = await client.call_tool("search", {"query": query, "top_k": payload.get("top_k",5)})
    ctx["memory"].add_short(ctx["session"]["session_id"], {"search_results": results})
    return {"status":"ok","results":results}

research_agent.register_skill("web_search", do_search)



search_node = TaskNode("search1","WebSearch","researcher","web_search",[],{"query":"latest AI models","top_k":3})

'''
When orchestrator executes that node:

The agent (ResearchAgent) receives message, calls the MCP tool, memory is updated, results captured.

Next nodes might process or summarize the results.
'''




