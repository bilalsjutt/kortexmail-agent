from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from typing import Optional, List, Dict, Any, Union
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import UpstashVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.tools import StructuredTool
from upstash_redis import Redis
import httpx
import json
import logging
import uuid
import asyncio
import itertools
from datetime import datetime
import os
from dotenv import load_dotenv
import tiktoken
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MCP AI Agent API",
    description="Serverless MCP AI Agent with Multi-Tool Support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm= ChatOpenAI(model='gpt-4o-mini',temperature=0.3)
# Pydantic models for API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Session ID")
    use_memory: bool = Field(default=True, description="Whether to use memory context")
    force_vector_search: bool = Field(default=False, description="Force vector store search")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: List[str] = []
    memory_used: Dict[str, Any] = {}


class MemoryStatus(BaseModel):
    redis_conversations: int
    window_size: int
    vector_store_status: str
    available_tools: int
    session_id: str


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class ToolCallResponse(BaseModel):
    result: str
    tool_name: str
    session_id: str
    success: bool


class AddServerRequest(BaseModel):
    name: str
    url: str


class UniversalMCPClient:
    """Universal MCP client that can handle multiple tool schemas and servers"""

    def __init__(self, servers_config: List[Dict[str, str]], timeout: int = 10):
        """
        Initialize with multiple MCP servers

        Args:
            servers_config: List of server configs with 'url' and optional 'name'
            timeout: Request timeout in seconds
        """
        self.servers = {}
        self.clients = {}
        self.request_counters = {}
        self.all_tools = {}
        self.timeout = timeout

        # Initialize the dictionaries BEFORE the loop
        for i, config in enumerate(servers_config):
            server_name = config.get('name', f'server_{i}')
            server_url = config['url']

            self.servers[server_name] = server_url

            # Now create the client AFTER server_name is defined
            self.clients[server_name] = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )

            self.request_counters[server_name] = itertools.count(1)

        logger.info(f"✅ Initialized Universal MCP Client with {len(self.servers)} servers")
    async def send_mcp_request(self, server_name: str, method: str, params: dict = None) -> dict:
        """Send MCP request to specific server"""
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

            response = await client.post(server_url, json=payload)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"MCP request failed for {server_name}: {e}")
            return {}

    # Update the discover_all_tools method
    async def discover_all_tools(self) -> Dict[str, Dict]:
        """Discover tools from all connected servers with timeout"""
        all_discovered_tools = {}

        for server_name, server_url in self.servers.items():
            try:
                logger.info(f"🔍 Discovering tools from {server_name}...")

                # Add timeout for each server
                try:
                    tools_response = await asyncio.wait_for(
                        self.send_mcp_request(server_name, "tools/list"),
                        timeout=10  # 10 second timeout per server
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout discovering tools from {server_name}")
                    continue

                if not tools_response:
                    logger.warning(f"No response from {server_name}")
                    continue

                # Rest of your parsing logic...
                tools_data = self._parse_tools_response(tools_response, server_name)

                if tools_data:
                    for tool_name, tool_info in tools_data.items():
                        tool_info['_server'] = server_name
                        tool_info['_server_url'] = server_url
                        all_discovered_tools[f"{server_name}_{tool_name}"] = tool_info

                    logger.info(f"✅ Found {len(tools_data)} tools from {server_name}")

            except Exception as e:
                logger.error(f"❌ Error discovering tools from {server_name}: {e}")
                continue  # Continue with other servers

        self.all_tools = all_discovered_tools
        logger.info(f"✅ Total discovered tools: {len(all_discovered_tools)}")
        return all_discovered_tools

    def _parse_tools_response(self, response: dict, server_name: str) -> Dict[str, Dict]:
        """Parse tools response handling different formats"""
        tools_data = {}

        try:
            # Format 1: {"result": {"tools": {...}}}
            if "result" in response and "tools" in response["result"]:
                raw_tools = response["result"]["tools"]

                # Handle both dict and list formats
                if isinstance(raw_tools, dict):
                    tools_data = raw_tools
                elif isinstance(raw_tools, list):
                    for tool in raw_tools:
                        if isinstance(tool, dict) and "name" in tool:
                            tools_data[tool["name"]] = tool

            # Format 2: {"tools": [...]} or {"tools": {...}}
            elif "tools" in response:
                raw_tools = response["tools"]
                if isinstance(raw_tools, list):
                    for tool in raw_tools:
                        if isinstance(tool, dict) and "name" in tool:
                            tools_data[tool["name"]] = tool
                elif isinstance(raw_tools, dict):
                    tools_data = raw_tools

            # Format 3: Direct list of tools
            elif isinstance(response, list):
                for tool in response:
                    if isinstance(tool, dict) and "name" in tool:
                        tools_data[tool["name"]] = tool

            # Normalize tool data
            normalized_tools = {}
            for tool_id, tool_info in tools_data.items():
                normalized_tool = {
                    "name": tool_info.get("name", tool_id),
                    "description": tool_info.get("description", f"Tool: {tool_id}"),
                    "inputSchema": tool_info.get("inputSchema", tool_info.get("input_schema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })),
                    "server": server_name
                }
                normalized_tools[tool_id] = normalized_tool

            return normalized_tools

        except Exception as e:
            logger.error(f"Error parsing tools response from {server_name}: {e}")
            return {}

    async def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """Call a tool, automatically routing to correct server"""
        try:
            # Find the tool and its server
            tool_info = None
            server_name = None

            # Check if tool name includes server prefix
            if tool_name in self.all_tools:
                tool_info = self.all_tools[tool_name]
                server_name = tool_info.get('_server')
            else:
                # Search for tool across all servers
                for full_tool_name, info in self.all_tools.items():
                    if info.get('name') == tool_name or full_tool_name.endswith(f"_{tool_name}"):
                        tool_info = info
                        server_name = info.get('_server')
                        break

            if not tool_info or not server_name:
                return f"Tool '{tool_name}' not found in any connected server"

            logger.info(f"🔧 Calling tool: {tool_name} on server: {server_name}")

            # Call the tool on appropriate server
            response_data = await self.send_mcp_request(server_name, "tools/call", {
                "name": tool_info.get('name', tool_name),
                "arguments": arguments or {}
            })

            if not response_data:
                return f"Error: Empty response when calling {tool_name}"

            # Handle error responses
            if "error" in response_data:
                error_info = response_data["error"]
                return f"MCP Error: {error_info.get('message', 'Unknown error')}"

            # Extract and format result
            result = response_data.get("result", response_data)
            return self._format_tool_result(tool_name, result)

        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format tool results based on tool type"""
        try:
            if isinstance(result, dict):
                # Special formatting for common tool types
                if "emails" in result and tool_name.endswith("get_emails"):
                    return self._format_emails_result(result)
                elif "files" in result:
                    return self._format_files_result(result)
                elif "data" in result:
                    return self._format_data_result(result)
                else:
                    return json.dumps(result, indent=2, default=str)
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error formatting result for {tool_name}: {e}")
            return str(result)

    def _format_emails_result(self, result: dict) -> str:
        """Format email results"""
        emails = result.get("emails", [])
        if not emails:
            return "No emails found."

        formatted = f"Found {len(emails)} emails:\n\n"
        for i, email in enumerate(emails[:5], 1):
            formatted += f"{i}. Subject: {email.get('subject', 'No subject')}\n"
            formatted += f"   From: {email.get('from', 'Unknown')}\n"
            formatted += f"   Category: {email.get('category', 'unknown')}\n"
            if email.get('starred'):
                formatted += f"   ⭐ Starred\n"
            body = email.get('body', '')
            if len(body) > 500:
                body = body[:500] + "..."
            formatted += f"   Preview: {body}\n\n"

        if len(emails) > 20:
            formatted += f"... and {len(emails) - 20} more emails"

        return formatted

    def _format_files_result(self, result: dict) -> str:
        """Format file results"""
        files = result.get("files", [])
        if not files:
            return "No files found."

        formatted = f"Found {len(files)} files:\n\n"
        for i, file in enumerate(files[:10], 1):
            formatted += f"{i}. {file.get('name', 'Unknown')}\n"
            if 'size' in file:
                formatted += f"   Size: {file['size']} bytes\n"
            if 'modified' in file:
                formatted += f"   Modified: {file['modified']}\n"

        return formatted

    def _format_data_result(self, result: dict) -> str:
        """Format generic data results"""
        data = result.get("data", [])
        if isinstance(data, list) and data:
            formatted = f"Found {len(data)} items:\n\n"
            for i, item in enumerate(data[:5], 1):
                if isinstance(item, dict):
                    # Show key fields
                    key_fields = ["name", "title", "id", "subject", "description"]
                    item_str = ""
                    for field in key_fields:
                        if field in item:
                            item_str += f"{field.title()}: {item[field]}, "
                    if item_str:
                        formatted += f"{i}. {item_str.rstrip(', ')}\n"
                    else:
                        formatted += f"{i}. {str(item)[:500]}...\n"
                else:
                    formatted += f"{i}. {str(item)[:500]}...\n"
            return formatted
        else:
            return json.dumps(result, indent=2, default=str)

    async def close(self):
        """Clean up all clients"""
        for client in self.clients.values():
            await client.aclose()


class SlidingWindowMemoryManager:
    def __init__(self, window_size: int = 10, max_tokens: int = 2000):
        self.window_size = window_size
        self.max_tokens = max_tokens

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Validate Redis credentials
        self.redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not self.redis_url or not self.redis_token:
            logger.warning("Redis credentials not found, memory features disabled")
            self.redis = None
            self.vector_store = None
            self.retriever = None
            return

        try:
            self.redis = Redis(
                url=self.redis_url,
                token=self.redis_token
            )

            # Test Redis connection asynchronously
            asyncio.create_task(self._test_redis_connection())

            # Initialize Vector Store
            vector_url = os.getenv("UPSTASH_VECTOR_REST_URL")
            vector_token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

            if not vector_url or not vector_token:
                logger.warning("Vector store credentials missing")
                self.vector_store = None
                self.retriever = None
                return

            self.embedding = OpenAIEmbeddings(model='text-embedding-3-small')
            self.vector_store = UpstashVectorStore(
                embedding=self.embedding,
                index_url=vector_url,
                index_token=vector_token,
                text_key="content"
            )

            # Setup retriever
            compressor = LLMChainExtractor.from_llm(llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
            )
        except Exception as e:
            logger.error(f"Memory manager initialization error: {e}")
            self.redis = None
            self.vector_store = None
            self.retriever = None

    async def _test_redis_connection(self):
        """Test Redis connection asynchronously"""
        try:
            if self.redis:
                # Use asyncio.to_thread for the sync Redis client
                await asyncio.wait_for(
                    asyncio.to_thread(self.redis.ping),
                    timeout=3
                )
                logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection test failed: {e}")
            self.redis = None  # Disable Redis if connection fails

    def get_session_key(self, session_id: str) -> str:
        """Get Redis key for session"""
        return f"chat_session_{session_id}"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    async def _summarize_conversation(self, human_msg: str, ai_msg: str) -> dict:
        """Create a summary when conversation exceeds token limit"""
        try:
            conversation_text = f"Human: {human_msg}\nAI: {ai_msg}"

            summary_prompt = f"""
            Please provide a concise summary of this conversation exchange that captures the key points and context:

            {conversation_text}

            Summary should be under 100 tokens and preserve important information for future reference.
            """

            response = await asyncio.to_thread(
                llm.invoke,
                [{"role": "user", "content": summary_prompt}]
            )

            summary = response.content.strip()

            return {
                "human": f"[SUMMARY] {human_msg[:50]}..." if len(human_msg) > 50 else human_msg,
                "ai": f"[SUMMARY] {summary}",
                "is_summary": True,
                "original_tokens": self._count_tokens(conversation_text)
            }

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            # Fallback: simple truncation
            return {
                "human": human_msg[:100] + "..." if len(human_msg) > 100 else human_msg,
                "ai": ai_msg[:200] + "..." if len(ai_msg) > 200 else ai_msg,
                "is_summary": True,
                "original_tokens": self._count_tokens(f"{human_msg}\n{ai_msg}")
            }

    async def add_conversation_to_memory(self, session_id: str, human_msg: str, ai_msg: str):
        """Add conversation with background vector processing and token management"""
        if not self.redis:
            logger.warning("Redis not available, skipping memory storage")
            return

        # Count tokens in the conversation
        conversation_text = f"Human: {human_msg}\nAI: {ai_msg}"
        token_count = self._count_tokens(conversation_text)

        # Decide whether to summarize or store full conversation
        if token_count > self.max_tokens:
            logger.info(f"Conversation exceeds {self.max_tokens} tokens ({token_count}), creating summary")
            summary_data = await self._summarize_conversation(human_msg, ai_msg)
            storage_human = summary_data["human"]
            storage_ai = summary_data["ai"]
            is_summary = True
        else:
            storage_human = human_msg
            storage_ai = ai_msg
            is_summary = False

        conversation_data = {
            "id": str(uuid.uuid4())[:8],
            "human": storage_human,
            "ai": storage_ai,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "is_summary": is_summary,
            "token_count": token_count if not is_summary else self._count_tokens(f"{storage_human}\n{storage_ai}")
        }

        try:
            # Create tasks for parallel execution
            redis_task = self._store_in_redis(session_id, conversation_data)

            # Only store in vector store if it's substantial
            vector_task = None
            if self.vector_store and self._is_substantial_exchange(human_msg, ai_msg):
                # For vector store, use original content even if Redis stores summary
                vector_conversation_data = {
                    **conversation_data,
                    "human": human_msg,  # Always use original for vector search
                    "ai": ai_msg
                }
                vector_task = self._store_in_vector_store(vector_conversation_data)

            # Execute Redis and Vector operations in parallel
            if vector_task:
                await asyncio.gather(
                    redis_task,
                    vector_task,
                    return_exceptions=True  # Don't fail if one operation fails
                )
            else:
                await redis_task

            logger.info(f"Successfully stored conversation for session {session_id} (summary: {is_summary})")

        except Exception as e:
            logger.error(f"Error in memory operations: {e}")

    async def _store_in_redis(self, session_id: str, conversation_data: dict):
        """Store conversation in Redis with proper error handling and parallel execution"""
        try:
            session_key = self.get_session_key(session_id)
            conversation_json = json.dumps(conversation_data)

            # Create all Redis operations as separate async tasks
            push_task = asyncio.to_thread(
                self.redis.lpush, session_key, conversation_json
            )

            trim_task = asyncio.to_thread(
                self.redis.ltrim, session_key, 0, self.window_size - 1
            )

            expire_task = asyncio.to_thread(
                self.redis.expire, session_key, 86400  # 24 hours
            )

            # Execute all operations in parallel with timeout
            await asyncio.wait_for(
                asyncio.gather(push_task, trim_task, expire_task),
                timeout=5
            )

            logger.debug(f"Successfully stored conversation in Redis for session {session_id}")

        except asyncio.TimeoutError:
            logger.error(f"Redis operations timed out for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error storing in Redis for session {session_id}: {e}")
            raise
    async def _store_in_vector_store(self, conversation_data: dict):
        """Store in vector store with timeout and error handling"""
        try:
            conversation_text = f"Human: {conversation_data['human']}\nAI: {conversation_data['ai']}"

            document = Document(
                page_content=conversation_text,
                metadata={
                    "conversation_id": conversation_data["id"],
                    "timestamp": conversation_data["timestamp"],
                    "session_id": conversation_data["session_id"],
                    "type": "conversation_exchange",
                    "is_summary": conversation_data.get("is_summary", False),
                    "token_count": conversation_data.get("token_count", 0)
                }
            )

            # Add timeout to vector operations
            await asyncio.wait_for(
                asyncio.to_thread(self.vector_store.add_documents, [document]),
                timeout=8  # Increased timeout for vector ops
            )

            logger.debug(f"Stored conversation {conversation_data['id']} in vector store")

        except asyncio.TimeoutError:
            logger.warning("Vector store operation timed out")
            # Don't raise - vector store is less critical than Redis
        except Exception as e:
            logger.error(f"Error storing in vector store: {e}")
            # Don't raise - vector store is less critical than Redis

    def _is_substantial_exchange(self, human_msg: str, ai_msg: str) -> bool:
        """Check if exchange is worth storing in vector store"""
        if len(human_msg.strip()) < 10 or len(ai_msg.strip()) < 20:
            return False

        trivial_patterns = {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "yes", "no", "sure", "bye", "goodbye", "status", "how are you"
        }

        human_lower = human_msg.lower().strip()
        return human_lower not in trivial_patterns and len(human_lower.split()) > 3

    async def get_session_memory_context(self, session_id: str) -> str:
        """Get memory context for session with improved formatting"""
        if not self.redis:
            return ""

        try:
            session_key = self.get_session_key(session_id)
            conversations_json = await asyncio.wait_for(
                asyncio.to_thread(self.redis.lrange, session_key, 0, -1),
                timeout=3  # Increased timeout
            )

            if not conversations_json:
                return ""

            context_parts = [f"[RECENT CONVERSATIONS - Session {session_id}]:"]
            total_tokens = 0

            for i, conv_json in enumerate(conversations_json):
                try:
                    conv_data = json.loads(conv_json)

                    # Add summary indicator if present
                    summary_indicator = " [SUMMARIZED]" if conv_data.get("is_summary", False) else ""

                    context_parts.append(f"\n{i + 1}.{summary_indicator} Human: {conv_data['human']}")
                    context_parts.append(f"   AI: {conv_data['ai']}")

                    # Track token usage
                    total_tokens += conv_data.get("token_count", 0)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse conversation JSON: {e}")
                    continue

            context = "\n".join(context_parts)
            logger.debug(f"Retrieved session memory: {len(conversations_json)} conversations, ~{total_tokens} tokens")

            return context

        except asyncio.TimeoutError:
            logger.error("Session memory retrieval timed out")
            return ""
        except Exception as e:
            logger.error(f"Error getting session memory: {e}")
            return ""

    async def search_vector_store(self, query: str, session_id: str = None) -> str:
        """Search vector store with improved parallel processing"""
        if not self.retriever:
            return ""

        try:
            # Create search task with timeout
            search_task = asyncio.wait_for(
                asyncio.to_thread(self.retriever.get_relevant_documents, query),
                timeout=8  # Increased timeout for complex searches
            )

            docs = await search_task

            if not docs:
                return ""

            # Filter by session if provided
            if session_id:
                docs = [doc for doc in docs if doc.metadata.get('session_id') == session_id]

            if not docs:
                return ""

            context_parts = [f"[RELEVANT PAST CONVERSATIONS]:"]
            for i, doc in enumerate(docs[:3], 1):  # Limit to top 3 results
                timestamp = doc.metadata.get('timestamp', 'unknown')
                is_summary = doc.metadata.get('is_summary', False)
                summary_indicator = " [SUMMARIZED]" if is_summary else ""

                context_parts.append(f"\n{i}.{summary_indicator} [{timestamp}]")
                context_parts.append(f"   {doc.page_content}")

            return "\n".join(context_parts)

        except asyncio.TimeoutError:
            logger.warning("Vector search timed out")
            return ""
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return ""

    def needs_vector_search(self, user_input: str) -> bool:
        """Determine if vector search is needed with improved keyword detection"""
        if not self.retriever:
            return False

        search_keywords = [
            'before', 'earlier', 'previous', 'what did', 'remember',
            'you said', 'we discussed', 'talked about', 'mentioned',
            'history', 'past', 'old conversation', 'when we', 'recall',
            'remind me', 'what was', 'did we', 'have we', 'previously'
        ]

        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in search_keywords)

    async def get_memory_stats(self, session_id: str) -> dict:
        """Get memory usage statistics for monitoring"""
        stats = {
            "redis_available": self.redis is not None,
            "vector_store_available": self.vector_store is not None,
            "session_conversations": 0,
            "total_tokens": 0,
            "summaries_count": 0
        }

        if not self.redis:
            return stats

        try:
            session_key = self.get_session_key(session_id)
            conversations_json = await asyncio.to_thread(self.redis.lrange, session_key, 0, -1)

            stats["session_conversations"] = len(conversations_json)

            for conv_json in conversations_json:
                try:
                    conv_data = json.loads(conv_json)
                    stats["total_tokens"] += conv_data.get("token_count", 0)
                    if conv_data.get("is_summary", False):
                        stats["summaries_count"] += 1
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")

        return stats


def create_langchain_tool_from_universal_mcp(tool_data: dict, mcp_client: UniversalMCPClient) -> StructuredTool:
    """Convert universal MCP tool to LangChain tool - Fixed for Pydantic v2"""

    # Create dynamic input schema based on tool's inputSchema
    input_schema = tool_data.get('inputSchema', {})
    properties = input_schema.get('properties', {})

    # Create field definitions for Pydantic v2
    field_definitions = {}

    for prop_name, prop_config in properties.items():
        # Map JSON schema types to Python types
        field_type = str  # Default to string
        field_default = None
        field_description = prop_config.get('description', f'{prop_name} parameter')

        if prop_config.get('type') == 'integer':
            field_type = int
        elif prop_config.get('type') == 'boolean':
            field_type = bool
        elif prop_config.get('type') == 'number':
            field_type = float
        elif prop_config.get('type') == 'array':
            field_type = List[str]  # Default array type
        elif prop_config.get('type') == 'object':
            field_type = Dict[str, Any]

        # Create properly annotated field with Optional wrapper
        field_definitions[prop_name] = (
            Optional[field_type],
            Field(default=field_default, description=field_description)
        )

    # If no properties defined, create a generic input
    if not field_definitions:
        field_definitions['query'] = (
            Optional[str],
            Field(default="", description="Input query or parameters")
        )

    # Use Pydantic's create_model function for dynamic model creation
    DynamicInput = create_model(
        f"{tool_data['name'].title()}Input",
        **field_definitions
    )

    async def tool_function(**kwargs) -> str:
        """Async tool function"""
        try:
            # Filter out None values
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = await mcp_client.call_tool(tool_data["name"], clean_kwargs)
            return str(result)
        except Exception as e:
            return f"Error calling {tool_data['name']}: {str(e)}"

    return StructuredTool.from_function(
        name=tool_data["name"],
        description=tool_data["description"],
        func=tool_function,
        args_schema=DynamicInput,
        coroutine=tool_function
    )


# Global instances (initialized on startup)
mcp_client = None
memory_manager = None
agent = None
tools = []


@app.on_event("startup")
async def startup_event():
    global mcp_client, memory_manager, agent, tools

    try:
        logger.info("🚀 Starting MCP AI Agent API...")

        # Add timeout and error handling for server discovery
        servers_config = [
            {"name": "email_server", "url": "https://kortexmail.vercel.app/api/mcp"},
        ]

        mcp_client = UniversalMCPClient(servers_config, timeout=10)  # 10 second timeout

        # Wrap tool discovery in timeout
        try:
            all_tools_data = await asyncio.wait_for(
                mcp_client.discover_all_tools(),
                timeout=10  # 10 second max for tool discovery
            )
        except asyncio.TimeoutError:
            logger.warning("Tool discovery timed out, continuing with empty tools")
            all_tools_data = {}

        # Continue even if no tools found
        tools = []
        for tool_name, tool_data in all_tools_data.items():
            try:
                lc_tool = create_langchain_tool_from_universal_mcp(tool_data, mcp_client)
                tools.append(lc_tool)
            except Exception as e:
                logger.error(f"❌ Error loading tool {tool_name}: {e}")
                continue  # Skip problematic tools

        # Initialize memory manager with validation
        try:
            memory_manager = SlidingWindowMemoryManager(window_size=10)
            logger.info("✅ Memory manager initialized")
        except Exception as e:
            logger.error(f"❌ Memory manager failed: {e}")
            # Continue without memory manager for basic functionality
            memory_manager = None

        # Initialize agent
        model = llm
        agent = create_react_agent(model, tools if tools else [])

        logger.info(f"✅ MCP AI Agent API started with {len(tools)} tools")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise - allow app to start in degraded mode
        logger.warning("Starting in degraded mode...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mcp_client
    if mcp_client:
        await mcp_client.close()
        logger.info("✅ MCP client closed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MCP AI Agent API",
        "status": "running",
        "tools_available": len(tools),
        "version": "1.0.0"
    }


@app.get("/tools")
async def list_tools():
    """List all available tools"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    tools_info = []
    for tool_name, tool_data in mcp_client.all_tools.items():
        tools_info.append({
            "name": tool_data.get("name"),
            "description": tool_data.get("description"),
            "server": tool_data.get("_server"),
            "input_schema": tool_data.get("inputSchema", {})
        })

    return {
        "tools": tools_info,
        "total": len(tools_info)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        logger.info(f"💬 Chat request from session {request.session_id}: {request.message}")

        # Get memory context if enabled AND memory_manager exists
        context_parts = []
        memory_info = {}

        if request.use_memory and memory_manager:  # Add null check
            # Get session memory
            session_context = await memory_manager.get_session_memory_context(request.session_id)
            if session_context:
                context_parts.append(session_context)
                memory_info["session_memory"] = "used"

            # Vector search if needed or forced
            if request.force_vector_search or memory_manager.needs_vector_search(request.message):
                vector_context = await memory_manager.search_vector_store(request.message, request.session_id)
                if vector_context:
                    context_parts.append(vector_context)
                    memory_info["vector_search"] = "used"



            # Vector search if needed or forced
            if request.force_vector_search or memory_manager.needs_vector_search(request.message):
                vector_context = await memory_manager.search_vector_store(request.message, request.session_id)
                if vector_context:
                    context_parts.append(vector_context)
                    memory_info["vector_search"] = "used"

        # Build prompt
        if context_parts:
            context_instruction = "\n[INSTRUCTION]: Use the conversation contexts above for accurate responses."
            full_prompt = f"[USER QUERY]: {request.message}\n\n{chr(10).join(context_parts)}{context_instruction}"
        else:
            full_prompt = request.message

        # Get response from agent
        response = await agent.ainvoke(
            {"messages": [("user", full_prompt)]},
            config={"recursion_limit": 10}
        )

        # Extract response
        if 'messages' in response and response['messages']:
            assistant_content = response['messages'][-1].content
        else:
            assistant_content = "I couldn't generate a response."

        # Store conversation if memory is enabled
        if request.use_memory:
            await memory_manager.add_conversation_to_memory(
                request.session_id,
                request.message,
                assistant_content
            )

        # Extract tools used (if any)
        tools_used = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tools_used = [call.get('name', 'unknown') for call in response.tool_calls]

        return ChatResponse(
            response=assistant_content,
            session_id=request.session_id,
            tools_used=tools_used,
            memory_used=memory_info
        )

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/tools/call", response_model=ToolCallResponse)
async def call_tool_directly(request: ToolCallRequest):
    """Direct tool calling endpoint"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    try:
        logger.info(f"🔧 Direct tool call: {request.tool_name}")

        result = await mcp_client.call_tool(request.tool_name, request.arguments)

        return ToolCallResponse(
            result=result,
            tool_name=request.tool_name,
            session_id=request.session_id,
            success=True
        )

    except Exception as e:
        logger.error(f"❌ Tool call error: {e}")
        return ToolCallResponse(
            result=f"Error: {str(e)}",
            tool_name=request.tool_name,
            session_id=request.session_id,
            success=False
        )



@app.get("/memory/status/{session_id}", response_model=MemoryStatus)
async def get_memory_status(session_id: str):
    """Get memory status for session"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        session_key = memory_manager.get_session_key(session_id)
        redis_count = await asyncio.to_thread(memory_manager.redis.llen, session_key)

        return MemoryStatus(
            redis_conversations=redis_count if redis_count else 0,
            window_size=memory_manager.window_size,
            vector_store_status="active",
            available_tools=len(tools),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"❌ Memory status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")



@app.delete("/memory/clear/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear memory for specific session"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        session_key = memory_manager.get_session_key(session_id)
        await asyncio.to_thread(memory_manager.redis.delete, session_key)

        return {
            "message": f"Memory cleared for session {session_id}",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"❌ Clear memory error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@app.post("/servers/add")
async def add_servers(request: AddServerRequest):
    """Add new MCP servers dynamically"""
    global mcp_client, agent, tools

    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    try:
        added_servers = []
        new_tools_count = 0

        for server_config in request.servers:
            if server_config.enabled:
                await mcp_client.add_server(server_config.name, server_config.url)
                added_servers.append(server_config.name)

        # Rebuild tools and agent
        all_tools_data = mcp_client.all_tools
        new_tools = []

        for tool_name, tool_data in all_tools_data.items():
            try:
                lc_tool = create_langchain_tool_from_universal_mcp(tool_data, mcp_client)
                new_tools.append(lc_tool)
            except Exception as e:
                logger.error(f"❌ Error loading tool {tool_name}: {e}")

        # Update global tools and recreate agent
        tools = new_tools
        model = llm
        agent = create_react_agent(model, tools)

        new_tools_count = len(tools) - len(added_servers)  # Approximate new tools

        return {
            "message": f"Added {len(added_servers)} servers",
            "added_servers": added_servers,
            "total_tools": len(tools),
            "new_tools": new_tools_count
        }

    except Exception as e:
        logger.error(f"❌ Add servers error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add servers: {str(e)}")


@app.delete("/servers/{server_name}")
async def remove_server(server_name: str):
    """Remove an MCP server"""
    global mcp_client, agent, tools

    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    try:
        if server_name not in mcp_client.servers:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

        await mcp_client.remove_server(server_name)

        # Rebuild tools and agent
        all_tools_data = mcp_client.all_tools
        new_tools = []

        for tool_name, tool_data in all_tools_data.items():
            try:
                lc_tool = create_langchain_tool_from_universal_mcp(tool_data, mcp_client)
                new_tools.append(lc_tool)
            except Exception as e:
                logger.error(f"❌ Error loading tool {tool_name}: {e}")

        # Update global tools and recreate agent
        tools = new_tools
        model = llm
        agent = create_react_agent(model, tools)

        return {
            "message": f"Removed server '{server_name}'",
            "remaining_servers": list(mcp_client.servers.keys()),
            "total_tools": len(tools)
        }

    except Exception as e:
        logger.error(f"❌ Remove server error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove server: {str(e)}")


@app.get("/servers")
async def list_servers():
    """List all connected MCP servers"""
    if not mcp_client:
        raise HTTPException(status_code=503, detail="MCP client not initialized")

    servers_info = []
    for name, url in mcp_client.servers.items():
        # Count tools from this server
        server_tools = [tool for tool, info in mcp_client.all_tools.items()
                        if info.get('_server') == name]

        servers_info.append({
            "name": name,
            "url": url,
            "tools_count": len(server_tools),
            "status": "connected"
        })

    return {
        "servers": servers_info,
        "total_servers": len(servers_info),
        "total_tools": len(mcp_client.all_tools)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "services": {
            "mcp_client": "ok" if mcp_client else "not_initialized",
            "memory_manager": "ok" if memory_manager else "not_initialized",
            "agent": "ok" if agent else "not_initialized"
        },
        "stats": {
            "servers_connected": len(mcp_client.servers) if mcp_client else 0,
            "tools_available": len(tools),
            "memory_window_size": memory_manager.window_size if memory_manager else 0
        }
    }

    # Check if any critical service is down
    if not all([mcp_client, memory_manager, agent]):
        health_status["status"] = "degraded"

    return health_status

@app.get("/ping")
async def ping():
    return {"status": "pong", "timestamp": datetime.now().isoformat()}

@app.post("/memory/search")
async def search_memory(
        query: str,
        session_id: Optional[str] = None,
        global_search: bool = False
):
    """Search conversation memory"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")

    try:
        search_session = None if global_search else session_id
        results = await memory_manager.search_vector_store(query, search_session)

        return {
            "query": query,
            "session_id": session_id,
            "global_search": global_search,
            "results": results if results else "No relevant conversations found",
            "has_results": bool(results)
        }

    except Exception as e:
        logger.error(f"❌ Memory search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/chat/batch")
async def chat_batch(requests: List[ChatRequest]):
    """Process multiple chat requests in batch"""
    if not agent or not memory_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")

    results = []

    for request in requests:
        try:
            # Process each request (simplified version of main chat endpoint)
            context_parts = []
            memory_info = {}

            if request.use_memory:
                session_context = await memory_manager.get_session_memory_context(request.session_id)
                if session_context:
                    context_parts.append(session_context)
                    memory_info["session_memory"] = "used"

            full_prompt = f"[USER QUERY]: {request.message}\n\n{chr(10).join(context_parts)}" if context_parts else request.message

            response = await agent.ainvoke(
                {"messages": [("user", full_prompt)]},
                config={"recursion_limit": 5}  # Lower limit for batch processing
            )

            assistant_content = response['messages'][-1].content if 'messages' in response and response[
                'messages'] else "No response generated"

            if request.use_memory:
                await memory_manager.add_conversation_to_memory(
                    request.session_id,
                    request.message,
                    assistant_content
                )

            results.append(ChatResponse(
                response=assistant_content,
                session_id=request.session_id,
                tools_used=[],
                memory_used=memory_info
            ))

        except Exception as e:
            logger.error(f"❌ Batch chat error for session {request.session_id}: {e}")
            results.append(ChatResponse(
                response=f"Error processing request: {str(e)}",
                session_id=request.session_id,
                tools_used=[],
                memory_used={}
            ))

    return {
        "results": results,
        "processed": len(results),
        "requested": len(requests)
    }


if __name__ == "__main__":
    import uvicorn

    # Set shorter timeouts for local testing
    os.environ.setdefault("STARTUP_TIMEOUT", "30")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=80,
        reload=True,
        log_level="info",
        timeout_keep_alive=30
    )
