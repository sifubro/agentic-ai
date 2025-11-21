# file: mcp_fs_server.py
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("FileSystem")

@mcp.tool()
def list_dir(path: str) -> list:
    return os.listdir(path)

@mcp.tool()
def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

@mcp.tool()
def write_file(path: str, content: str) -> bool:
    with open(path, 'w') as f:
        f.write(content)
    return True

if __name__ == "__main__":
    mcp.run(transport="stdio")
