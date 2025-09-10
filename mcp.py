from langsmith import traceable
from typing import Optional, List, Dict, Any, Union
import logging
import asyncio
import httpx
import json
import itertools
from pydantic import Field, create_model
from langchain_core.tools import StructuredTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalMCPClient:
    """OPTIMIZED: Universal MCP client with performance improvements"""

    def __init__(self, servers_config: List[Dict[str, str]], timeout: int = 8):  # OPTIMIZED: Reduced timeout
        """Initialize with faster defaults"""
        self.servers = {}
        self.clients = {}
        self.request_counters = {}
        self.all_tools = {}
        self.timeout = timeout

        # OPTIMIZED: Pre-create connection pools
        for i, config in enumerate(servers_config):
            server_name = config.get('name', f'server_{i}')
            server_url = config['url']
            self.servers[server_name] = server_url

            # OPTIMIZED: Smaller connection limits for faster startup
            self.clients[server_name] = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
            self.request_counters[server_name] = itertools.count(1)

        logger.info(f"Initialized MCP Client with {len(self.servers)} servers")

    @traceable(name="mcp_send_request")
    async def send_mcp_request(self, server_name: str, method: str, params: dict = None) -> dict:
        """OPTIMIZED: Faster MCP request with better error handling"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")

        payload = {
            "jsonrpc": "2.0",
            "id": next(self.request_counters[server_name]),
            "method": method,
            "params": params or {}
        }

        try:
            client = self.clients[server_name]
            server_url = self.servers[server_name]

            # OPTIMIZED: Single request with timeout
            response = await asyncio.wait_for(
                client.post(server_url, json=payload),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {server_name} - {method}")
            return {}
        except Exception as e:
            logger.error(f"MCP request failed for {server_name}: {e}")
            return {}

    @traceable(name="mcp_discover_tools")
    async def discover_all_tools(self) -> Dict[str, Dict]:
        """OPTIMIZED: Parallel tool discovery with aggressive timeouts"""
        all_discovered_tools = {}

        # OPTIMIZED: Create all discovery tasks concurrently
        discovery_tasks = []
        for server_name, server_url in self.servers.items():
            task = asyncio.create_task(
                self._discover_server_tools(server_name, server_url)
            )
            discovery_tasks.append(task)

        # OPTIMIZED: Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*discovery_tasks, return_exceptions=True),
                timeout=15  # Total discovery timeout
            )

            for server_name, result in zip(self.servers.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Tool discovery failed for {server_name}: {result}")
                    continue

                if result:
                    for tool_name, tool_info in result.items():
                        tool_info['_server'] = server_name
                        tool_info['_server_url'] = self.servers[server_name]
                        all_discovered_tools[f"{server_name}_{tool_name}"] = tool_info

                    logger.info(f"Found {len(result)} tools from {server_name}")

        except asyncio.TimeoutError:
            logger.error("Overall tool discovery timed out")

        self.all_tools = all_discovered_tools
        logger.info(f"Total discovered tools: {len(all_discovered_tools)}")
        return all_discovered_tools

    async def _discover_server_tools(self, server_name: str, server_url: str) -> Dict[str, Dict]:
        """OPTIMIZED: Single server tool discovery with timeout"""
        try:
            logger.info(f"Discovering tools from {server_name}...")

            tools_response = await asyncio.wait_for(
                self.send_mcp_request(server_name, "tools/list"),
                timeout=8  # Per-server timeout
            )

            if not tools_response:
                logger.warning(f"No response from {server_name}")
                return {}

            tools_data = self._parse_tools_response(tools_response, server_name)
            return tools_data

        except asyncio.TimeoutError:
            logger.warning(f"Timeout discovering tools from {server_name}")
            return {}
        except Exception as e:
            logger.error(f"Error discovering tools from {server_name}: {e}")
            return {}

    def _parse_tools_response(self, response: dict, server_name: str) -> Dict[str, Dict]:
        """OPTIMIZED: Faster tools response parsing"""
        tools_data = {}

        try:
            # OPTIMIZED: Direct pattern matching for common formats
            raw_tools = None

            if "result" in response and "tools" in response["result"]:
                raw_tools = response["result"]["tools"]
            elif "tools" in response:
                raw_tools = response["tools"]
            elif isinstance(response, list):
                raw_tools = response

            if not raw_tools:
                return {}

            # OPTIMIZED: Handle list/dict formats efficiently
            if isinstance(raw_tools, list):
                for tool in raw_tools:
                    if isinstance(tool, dict) and "name" in tool:
                        tools_data[tool["name"]] = self._normalize_tool(tool, server_name)
            elif isinstance(raw_tools, dict):
                for tool_id, tool_info in raw_tools.items():
                    if isinstance(tool_info, dict):
                        tools_data[tool_id] = self._normalize_tool(tool_info, server_name)

            return tools_data

        except Exception as e:
            logger.error(f"Error parsing tools response from {server_name}: {e}")
            return {}

    def _normalize_tool(self, tool_info: dict, server_name: str) -> dict:
        """OPTIMIZED: Fast tool normalization"""
        return {
            "name": tool_info.get("name", "unknown"),
            "description": tool_info.get("description", f"Tool from {server_name}"),
            "inputSchema": tool_info.get("inputSchema", tool_info.get("input_schema", {
                "type": "object",
                "properties": {},
                "required": []
            })),
            "server": server_name
        }

    @traceable(name="mcp_call_tool")
    async def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """OPTIMIZED: Faster tool calling with caching"""
        try:
            # OPTIMIZED: Direct lookup first
            tool_info = self.all_tools.get(tool_name)
            server_name = None

            if tool_info:
                server_name = tool_info.get('_server')
            else:
                # OPTIMIZED: Faster search through tools
                for full_name, info in self.all_tools.items():
                    if (info.get('name') == tool_name or
                            full_name.endswith(f"_{tool_name}")):
                        tool_info = info
                        server_name = info.get('_server')
                        break

            if not tool_info or not server_name:
                return f"Tool '{tool_name}' not found"

            logger.info(f"Calling tool: {tool_name} on {server_name}")

            # OPTIMIZED: Direct tool call with timeout
            response_data = await asyncio.wait_for(
                self.send_mcp_request(server_name, "tools/call", {
                    "name": tool_info.get('name', tool_name),
                    "arguments": arguments or {}
                }),
                timeout=20  # Tool execution timeout
            )

            if not response_data:
                return f"Empty response from {tool_name}"

            # OPTIMIZED: Simple error handling
            if "error" in response_data:
                return f"MCP Error: {response_data['error'].get('message', 'Unknown error')}"

            result = response_data.get("result", response_data)
            return self._format_tool_result(tool_name, result)

        except asyncio.TimeoutError:
            return f"Tool {tool_name} timed out"
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return f"Error calling {tool_name}: {str(e)}"

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        """OPTIMIZED: Faster result formatting with email focus"""
        try:
            if isinstance(result, dict):
                # OPTIMIZED: Quick email detection and formatting
                if "emails" in result:
                    return json.dumps(result, default=str)  # Let agent handle extraction
                elif "data" in result:
                    return json.dumps(result, default=str)
                else:
                    return json.dumps(result, indent=2, default=str)
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error formatting result for {tool_name}: {e}")
            return str(result)

    async def close(self):
        """OPTIMIZED: Faster cleanup"""
        close_tasks = [client.aclose() for client in self.clients.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("MCP Client closed")


def create_langchain_tool_from_universal_mcp(tool_data: dict, mcp_client: UniversalMCPClient) -> StructuredTool:
    """OPTIMIZED: Faster LangChain tool creation with simpler schemas"""

    input_schema = tool_data.get('inputSchema', {})
    properties = input_schema.get('properties', {})

    # OPTIMIZED: Simplified field creation
    field_definitions = {}

    for prop_name, prop_config in properties.items():
        field_type = str  # Default to string for simplicity
        field_description = prop_config.get('description', f'{prop_name} parameter')

        # OPTIMIZED: Basic type mapping only
        prop_type = prop_config.get('type', 'string')
        if prop_type == 'integer':
            field_type = int
        elif prop_type == 'boolean':
            field_type = bool

        field_definitions[prop_name] = (
            Optional[field_type],
            Field(default=None, description=field_description)
        )

    # OPTIMIZED: Always include a default field
    if not field_definitions:
        field_definitions['query'] = (
            Optional[str],
            Field(default="", description="Query parameter")
        )

    # OPTIMIZED: Create dynamic model
    try:
        DynamicInput = create_model(
            f"{tool_data['name'].replace('-', '_').title()}Input",
            **field_definitions
        )
    except Exception as e:
        logger.warning(f"Schema creation failed for {tool_data['name']}: {e}")
        # Fallback to generic input
        DynamicInput = create_model(
            f"GenericInput",
            query=(Optional[str], Field(default="", description="Input parameter"))
        )

    async def tool_function(**kwargs) -> str:
        """OPTIMIZED: Faster tool execution"""
        try:
            # OPTIMIZED: Filter None values and execute
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = await mcp_client.call_tool(tool_data["name"], clean_kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return StructuredTool.from_function(
        name=tool_data["name"],
        description=tool_data["description"][:200],  # OPTIMIZED: Limit description length
        func=tool_function,
        args_schema=DynamicInput,
        coroutine=tool_function
    )