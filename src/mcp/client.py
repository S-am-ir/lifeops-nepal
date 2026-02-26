from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from typing import List, Dict
from src.config.settings import settings

MCP_SERVERS: Dict = {
    "travel": {
        "url": f"http://127.0.0.1:{settings.mcp_travel_port}/mcp",
        "transport": "streamable_http"
    },
    "comms": {
        "url": f"http://127.0.0.1:{settings.mcp_comms_port}/mcp",
        "transport": "streamable_http"
    },
    "moodboard": {
        "url": f"http://127.0.0.1:{settings.mcp_moodbboard_port}/mcp",
        "transport": "streamable_http"
    } 
}

_client: MultiServerMCPClient | None = None
_tools: List[BaseTool] = []

async def get_mcp_tools(servers: List[str] | None = None) -> List[BaseTool]:
    global _client, _tools
    if not _tools:
        target = {k: v for k, v in MCP_SERVERS.items() if servers is None or k in servers}
        _client = MultiServerMCPClient(target)
        await _client.__aenter__()
        _tools = await _client.get_tools()
        print(f"[MCP] Loaded {len(_tools)} tools: {[t.name for t in _tools]}")
    if servers:
        # Filter the requested server tools if a subset was asked for 
        return [t for t in _tools if any(s in t.name for s in servers)] 
    
    return _tools

async def reset_mcp_client():
    global _client, _tools
    if _client:
        try:
            await _client.__aexit__(None, None, None)
        except Exception as e:
            print(f"[MCP] Cleanup warning: {e}")
    _client = None
    _tools = []
    return await get_mcp_tools()

