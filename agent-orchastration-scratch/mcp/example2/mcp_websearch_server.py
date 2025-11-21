# file: mcp_websearch_server.py
from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("WebSearch")

@mcp.tool()
def search(query: str, top_k: int = 5) -> list:
    # simplistic: call a public search API
    resp = requests.get("https://api.example.com/search", params={"q": query, "k": top_k})
    return resp.json()["results"]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=9000)
