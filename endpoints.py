import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Dict, Any
import asyncio
import os
from dotenv import load_dotenv

# Import the agent manager from the updated agent module
from src.agent import agent_manager
from src.response_handler import (
    ChatRequest, ChatResponse, MemoryStatus, ToolCallRequest,
    ToolCallResponse, AddServerRequest, StreamEvent, HealthStatus
)

# Load environment variables like old code
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up LangSmith environment like old code
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "pr-brief-recapitulation-91")

# LangSmith imports like old code
from langsmith import traceable, Client
from langchain import globals

# Enable LangSmith tracing globally like old code
globals.set_debug(True)
globals.set_verbose(True)

# Initialize LangSmith client like old code
try:
    langsmith_client = Client()
    logger.info("LangSmith client initialized")
except Exception as e:
    logger.warning(f"LangSmith client initialization failed: {e}")
    langsmith_client = None

app = FastAPI(
    title="Email AI Agent API",
    description="Email AI Agent with MCP tool integration",
    version="2.0.0"
)

# Default MCP servers configuration - using the working URL from old code
DEFAULT_SERVERS = [
    {"name": "email_server", "url": "https://kortexmail.vercel.app/api/mcp"},
]

# Add CORS middleware like old code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event - using old code's simple initialization pattern
@app.on_event("startup")
async def startup_event():
    """Startup event handler - using old code pattern"""
    try:
        logger.info("Starting Email AI Agent...")

        # Initialize agent manager with MCP client like old code
        await agent_manager.initialize_mcp_client(DEFAULT_SERVERS)

        logger.info(f"Started with {len(agent_manager.tools)} tools")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue startup even if MCP fails, like old code


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    try:
        logger.info("Shutting down Email AI Agent...")
        await agent_manager.cleanup()
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Health and Status Endpoints
@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def get_health_status():
    """Get overall system health status"""
    try:
        return await agent_manager.get_health_status()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Email AI Agent API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


# Chat Endpoints - using old code's streaming pattern
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Process a chat request"""
    try:
        logger.info(f"Processing chat request for session: {request.session_id}")
        response = await agent_manager.process_chat_request(request)
        logger.info(f"Chat response generated for session: {request.session_id}")
        return response
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@traceable(name="chat_stream_endpoint")
@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(request: ChatRequest):
    """Stream a chat response - using old code's streaming logic"""
    try:
        logger.info(f"Starting stream chat for session: {request.session_id}")

        async def generate_stream():
            try:
                async for chunk in agent_manager.stream_chat_response(request):
                    yield chunk
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                error_data = {"error": str(e), "status": "error"}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        logger.error(f"Stream chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}")


# Tool Management Endpoints
@app.get("/tools", tags=["Tools"])
async def list_tools():
    """List all available MCP tools"""
    try:
        tools = await agent_manager.list_available_tools()
        logger.info(f"Listed {len(tools)} available tools")
        return {"tools": tools, "count": len(tools)}
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tools")


@app.post("/tools/call", response_model=ToolCallResponse, tags=["Tools"])
async def call_tool(request: ToolCallRequest):
    """Call a specific tool directly"""
    try:
        logger.info(f"Calling tool: {request.tool_name}")
        response = await agent_manager.call_tool_directly(request)

        if response.success:
            logger.info(f"Tool {request.tool_name} executed successfully")
        else:
            logger.warning(f"Tool {request.tool_name} execution failed")

        return response
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


# Debug endpoints - using old code pattern
@app.get("/debug/tools")
async def debug_tools():
    """Debug endpoint to see loaded tools - from old code"""
    return {
        "tools_count": len(agent_manager.tools),
        "tool_names": [tool.name for tool in agent_manager.tools],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
            } for tool in agent_manager.tools
        ],
        "mcp_client_status": "connected" if agent_manager.mcp_client else "not_connected"
    }


@app.get("/debug/mcp")
async def debug_mcp():
    """Debug MCP connection - from old code"""
    try:
        if not agent_manager.mcp_client:
            return {"error": "MCP client not initialized"}

        test_result = await agent_manager.mcp_client.discover_all_tools()
        return {
            "mcp_status": "connected",
            "discovered_tools": list(test_result.keys()),
            "server_url": "https://kortexmail.vercel.app/api/mcp"
        }
    except Exception as e:
        return {"error": str(e), "mcp_status": "failed"}


@app.post("/tools/discover", tags=["Tools"])
async def rediscover_tools():
    """Rediscover tools from all connected MCP servers"""
    try:
        logger.info("Rediscovering tools from MCP servers...")
        await agent_manager._discover_and_setup_tools()
        tools = await agent_manager.list_available_tools()
        logger.info(f"Rediscovered {len(tools)} tools")
        return {
            "message": "Tools rediscovered successfully",
            "tools_found": len(tools)
        }
    except Exception as e:
        logger.error(f"Tool rediscovery failed: {e}")
        raise HTTPException(status_code=500, detail="Tool rediscovery failed")


# Memory Management Endpoints
@app.get("/memory/{session_id}", response_model=MemoryStatus, tags=["Memory"])
async def get_memory_status(session_id: str):
    """Get memory status for a session"""
    try:
        logger.info(f"Getting memory status for session: {session_id}")
        status = await agent_manager.get_memory_status(session_id)
        return status
    except Exception as e:
        logger.error(f"Memory status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Memory status retrieval failed")


@app.delete("/memory/{session_id}", tags=["Memory"])
async def clear_session_memory(session_id: str):
    """
    Clear all memory for a specific session

    - **session_id**: Session identifier to clear memory for
    """
    try:
        logger.info(f"🗑️ Clearing memory for session: {session_id}")

        if agent_manager.memory_manager.redis:
            session_key = agent_manager.memory_manager.get_session_key(session_id)
            await asyncio.to_thread(agent_manager.memory_manager.redis.delete, session_key)
            logger.info(f"✅ Memory cleared for session: {session_id}")
            return {"message": f"Memory cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=503, detail="Memory system not available")

    except Exception as e:
        logger.error(f"❌ Memory clearing failed: {e}")
        raise HTTPException(status_code=500, detail="Memory clearing failed")


@app.get("/memory/stats", tags=["Memory"])
async def get_global_memory_stats():
    """
    Get global memory system statistics
    """
    try:
        if not agent_manager.memory_manager.redis:
            return {"message": "Memory system not available"}

        # Get basic Redis info
        info = await asyncio.to_thread(agent_manager.memory_manager.redis.info)

        return {
            "redis_available": True,
            "vector_store_available": agent_manager.memory_manager.vector_store is not None,
            "redis_memory_usage": info.get("used_memory_human", "unknown"),
            "redis_connected_clients": info.get("connected_clients", 0),
            "uptime_seconds": info.get("uptime_in_seconds", 0)
        }
    except Exception as e:
        logger.error(f"❌ Global memory stats failed: {e}")
        raise HTTPException(status_code=500, detail="Memory stats retrieval failed")


# Server Management Endpoints
@app.post("/servers", tags=["Servers"])
async def add_mcp_server(request: AddServerRequest):
    """
    Add a new MCP server to the system

    - **name**: Unique name for the server
    - **url**: MCP server URL endpoint
    """
    try:
        logger.info(f"➕ Adding MCP server: {request.name} at {request.url}")
        result = await agent_manager.add_server(request)
        logger.info(f"✅ Server {request.name} added successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Server addition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server addition failed: {str(e)}")


@app.get("/servers", tags=["Servers"])
async def list_mcp_servers():
    """
    List all connected MCP servers
    """
    try:
        if not agent_manager.mcp_client:
            return {"servers": [], "count": 0}

        servers = [
            {"name": name, "url": url}
            for name, url in agent_manager.mcp_client.servers.items()
        ]

        return {"servers": servers, "count": len(servers)}
    except Exception as e:
        logger.error(f"❌ Server listing failed: {e}")
        raise HTTPException(status_code=500, detail="Server listing failed")


@app.delete("/servers/{server_name}", tags=["Servers"])
async def remove_mcp_server(server_name: str):
    """
    Remove an MCP server from the system

    - **server_name**: Name of the server to remove
    """
    try:
        if not agent_manager.mcp_client or server_name not in agent_manager.mcp_client.servers:
            raise HTTPException(status_code=404, detail="Server not found")

        logger.info(f"➖ Removing MCP server: {server_name}")

        # Remove server and client
        del agent_manager.mcp_client.servers[server_name]
        if server_name in agent_manager.mcp_client.clients:
            await agent_manager.mcp_client.clients[server_name].aclose()
            del agent_manager.mcp_client.clients[server_name]

        # Rediscover tools
        await agent_manager._discover_and_setup_tools()

        logger.info(f"✅ Server {server_name} removed successfully")
        return {"message": f"Server {server_name} removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Server removal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Server removal failed: {str(e)}")


# Development and Testing Endpoints
@app.post("/dev/test-memory", tags=["Development"])
async def test_memory_system(session_id: str = "test_session"):
    """
    Test memory system functionality (development only)
    """
    try:
        logger.info("🧪 Testing memory system...")

        # Test storing a conversation
        await agent_manager.memory_manager.add_conversation_to_memory(
            session_id,
            "This is a test message",
            "This is a test response from the AI"
        )

        # Test retrieving memory
        context = await agent_manager.memory_manager.get_session_memory_context(session_id)

        return {
            "memory_stored": True,
            "context_retrieved": bool(context),
            "context_length": len(context),
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory test failed: {str(e)}")


@app.post("/dev/test-tools", tags=["Development"])
async def test_tool_discovery():
    """
    Test tool discovery from MCP servers (development only)
    """
    try:
        logger.info("🧪 Testing tool discovery...")

        if not agent_manager.mcp_client:
            raise HTTPException(status_code=503, detail="MCP client not initialized")

        discovered = await agent_manager.mcp_client.discover_all_tools()

        return {
            "discovery_successful": True,
            "tools_discovered": len(discovered),
            "servers_checked": len(agent_manager.mcp_client.servers),
            "tool_names": list(discovered.keys())
        }
    except Exception as e:
        logger.error(f"❌ Tool discovery test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool discovery test failed: {str(e)}")


# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint was not found",
        "status_code": 404
    }


# Add this to your endpoints.py file

@app.post("/dev/test-email-tools", tags=["Development"])
async def test_email_tool_extraction():
    """Test email tool calling and artifact extraction"""
    try:
        logger.info("Testing email tool execution...")

        # Find email tools
        email_tools = []
        for tool in agent_manager.tools:
            tool_name_lower = tool.name.lower()
            if any(keyword in tool_name_lower for keyword in
                   ['email', 'get_emails', 'search_email', 'find_email', 'fetch_email']):
                email_tools.append(tool)

        if not email_tools:
            return {"error": "No email tools found", "available_tools": [t.name for t in agent_manager.tools]}

        # Test the first email tool
        test_tool = email_tools[0]
        logger.info(f"Testing tool: {test_tool.name}")

        # Call the tool directly
        try:
            result = await test_tool.arun({})
            logger.info(f"Tool result type: {type(result)}")
            logger.info(f"Tool result preview: {str(result)[:500]}...")

            # Test artifact extraction
            artifacts = agent_manager._extract_email_artifacts(result)

            return {
                "tool_tested": test_tool.name,
                "tool_result_type": str(type(result)),
                "tool_result_length": len(str(result)),
                "tool_result_preview": str(result)[:200],
                "artifacts_extracted": len(artifacts),
                "artifacts": artifacts[:3] if artifacts else [],  # Show first 3
                "success": True
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "tool_tested": test_tool.name,
                "error": str(e),
                "success": False
            }

    except Exception as e:
        logger.error(f"Email tool test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Email tool test failed: {str(e)}")


@app.get("/dev/debug-stream-response", tags=["Development"])
async def debug_stream_components():
    """Debug the stream response components"""
    try:
        # Check tool identification
        email_tools = []
        all_tools = []

        for tool in agent_manager.tools:
            all_tools.append(tool.name)
            tool_name_lower = tool.name.lower()
            if any(keyword in tool_name_lower for keyword in
                   ['email', 'get_emails', 'search_email', 'find_email', 'fetch_email']):
                email_tools.append(tool.name)

        return {
            "total_tools": len(agent_manager.tools),
            "all_tools": all_tools,
            "identified_email_tools": email_tools,
            "mcp_client_status": "connected" if agent_manager.mcp_client else "not_connected",
            "mcp_servers": list(agent_manager.mcp_client.servers.keys()) if agent_manager.mcp_client else []
        }

    except Exception as e:
        logger.error(f"Debug stream components failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )