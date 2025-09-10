import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable
import json
import asyncio
import logging
from typing import List, Dict, Any, AsyncGenerator
import uuid
from datetime import datetime
import os

from .agent_memory import MemoryManager
from .mcp import UniversalMCPClient, create_langchain_tool_from_universal_mcp
from .response_handler import (
    ChatRequest, ChatResponse, MemoryStatus, ToolCallRequest,
    ToolCallResponse, AddServerRequest, StreamEvent, HealthStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OPTIMIZED: Simplified system prompt - no JSON formatting required
SYSTEM_PROMPT_TEMPLATE = """You are an intelligent email assistant that helps users manage their emails efficiently.

ROLE & BEHAVIOR:
- Be concise and helpful
- Use tools when email data is requested
- Provide clear, actionable responses

AVAILABLE TOOLS:
{tool_list}

MEMORY CONTEXT:
{memory_context}

Always use appropriate tools when users ask about emails. Be direct and helpful in your responses."""


class AgentManager:
    def __init__(self):
        # OPTIMIZED: Reduced temperature and model for faster responses
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            streaming=True,
            temperature=0.1,  # Reduced for faster, more consistent responses
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=500  # OPTIMIZED: Limit response length
        )
        self.memory_manager = MemoryManager(window_size=5)  # OPTIMIZED: Reduced window
        self.mcp_client = None
        self.tools = []
        self._start_time = datetime.now()

        # OPTIMIZED: Cache email tools for faster lookup
        self._email_tools_cache = set()

    async def initialize_mcp_client(self, servers_config: List[Dict[str, str]]):
        """OPTIMIZED: Faster MCP client initialization"""
        try:
            logger.info("Starting MCP Client initialization...")

            # OPTIMIZED: Reduced timeout for faster startup
            self.mcp_client = UniversalMCPClient(servers_config, timeout=8)

            # OPTIMIZED: Parallel tool discovery with shorter timeout
            try:
                all_tools_data = await asyncio.wait_for(
                    self.mcp_client.discover_all_tools(),
                    timeout=8  # Reduced timeout
                )
                logger.info(f"Discovered tools: {list(all_tools_data.keys())}")
            except asyncio.TimeoutError:
                logger.warning("Tool discovery timed out")
                all_tools_data = {}

            # OPTIMIZED: Cache tool creation
            self.tools = []
            self._email_tools_cache.clear()

            for tool_name, tool_data in all_tools_data.items():
                try:
                    lc_tool = create_langchain_tool_from_universal_mcp(tool_data, self.mcp_client)
                    self.tools.append(lc_tool)

                    # OPTIMIZED: Cache email tools
                    if any(keyword in tool_name.lower() for keyword in
                           ['email', 'get_emails', 'search_email', 'find_email']):
                        self._email_tools_cache.add(tool_name)

                    logger.info(f"Loaded tool: {tool_name} -> {lc_tool.name}")
                except Exception as e:
                    logger.error(f"Error loading tool {tool_name}: {e}")

            logger.info(
                f"MCP Client initialized with {len(self.tools)} tools, {len(self._email_tools_cache)} email tools")

        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self.tools = []
            self._email_tools_cache.clear()

    def _get_system_prompt(self, memory_context: str = "", session_id: str = "") -> str:
        """OPTIMIZED: Simplified system prompt generation"""
        tool_list = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_list=tool_list,
            memory_context=memory_context if memory_context else "No previous context."
        )

    @traceable(name="stream_chat_response")
    async def stream_chat_response(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """OPTIMIZED: Streamlined chat response with faster tool processing"""
        try:
            # OPTIMIZED: Skip memory for simple requests to improve speed
            memory_context = ""
            if request.use_memory and len(request.message.split()) > 5:  # Only use memory for complex queries
                try:
                    memory_context = await asyncio.wait_for(
                        self.memory_manager.get_session_memory_context(
                            request.session_id, request.message, self.llm
                        ),
                        timeout=2  # OPTIMIZED: Shorter memory timeout
                    )
                except (Exception, asyncio.TimeoutError) as e:
                    logger.warning(f"Memory context skipped: {e}")

            system_prompt = self._get_system_prompt(memory_context, request.session_id)
            llm_with_tools = self.llm.bind_tools(self.tools)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]

            # OPTIMIZED: Track tool execution state
            response_buffer = ""
            tools_executed = []
            tool_results = {}
            tool_dict = {tool.name: tool for tool in self.tools}

            # OPTIMIZED: Pre-determine if this looks like an email request
            is_email_request = any(keyword in request.message.lower()
                                   for keyword in ['email', 'inbox', 'message', 'mail'])

            logger.info(f"Processing request: {request.message[:50]}... (email_request: {is_email_request})")

            # OPTIMIZED: Stream with concurrent tool execution
            tool_execution_tasks = {}

            async for chunk in llm_with_tools.astream(messages):
                # Handle tool calls with immediate execution
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})

                        if tool_name not in tools_executed:
                            tools_executed.append(tool_name)

                            # Send thinking event
                            yield self._create_stream_event("thinking",
                                                            {'status': f'executing {tool_name.replace("_", " ")}'})

                            # OPTIMIZED: Execute tool immediately without waiting
                            if tool_name in tool_dict:
                                task = asyncio.create_task(
                                    self._execute_tool_safe(tool_dict[tool_name], tool_args, tool_name)
                                )
                                tool_execution_tasks[tool_name] = task

                # Handle content
                elif hasattr(chunk, 'content') and chunk.content:
                    response_buffer += chunk.content

            # OPTIMIZED: Wait for all tool executions to complete
            if tool_execution_tasks:
                logger.info(f"Waiting for {len(tool_execution_tasks)} tools to complete...")
                completed_results = await asyncio.gather(*tool_execution_tasks.values(), return_exceptions=True)

                for i, (tool_name, task) in enumerate(tool_execution_tasks.items()):
                    result = completed_results[i]
                    if isinstance(result, Exception):
                        tool_results[tool_name] = f"Error: {str(result)}"
                        logger.error(f"Tool {tool_name} failed: {result}")
                    else:
                        tool_results[tool_name] = result
                        logger.info(f"Tool {tool_name} completed successfully")

            # OPTIMIZED: Process results immediately
            artifacts = []
            final_message = response_buffer.strip() or "I've processed your request."

            # OPTIMIZED: Extract artifacts from email tool results
            if tool_results:
                logger.info(f"Processing {len(tool_results)} tool results for artifacts...")

                for tool_name, result in tool_results.items():
                    # Check if this tool might contain email data
                    if (tool_name in self._email_tools_cache or
                            any(keyword in tool_name.lower() for keyword in ['email', 'mail'])):
                        try:
                            extracted_artifacts = self._extract_email_artifacts(result)
                            if extracted_artifacts:
                                artifacts.extend(extracted_artifacts)
                                logger.info(f"Extracted {len(extracted_artifacts)} artifacts from {tool_name}")
                        except Exception as e:
                            logger.error(f"Artifact extraction failed for {tool_name}: {e}")

            # Send response
            if final_message:
                yield self._create_stream_event("message", {'content': final_message})

            # OPTIMIZED: Send artifacts in single batch
            if artifacts:
                logger.info(f"Sending {len(artifacts)} artifacts in batch")
                # Add metadata to artifacts
                for i, artifact in enumerate(artifacts):
                    artifact.update({
                        'timestamp': datetime.now().isoformat(),
                        'session_id': request.session_id,
                        'artifact_index': i
                    })

                # Send all artifacts at once
                yield self._create_stream_event("artifacts", {'data': artifacts})
            else:
                logger.debug("No artifacts to send")

            # OPTIMIZED: Background memory storage
            if request.use_memory and final_message:
                asyncio.create_task(
                    self.memory_manager.add_conversation_to_memory(
                        request.session_id, request.message, final_message
                    )
                )

            # Send completion
            yield self._create_stream_event("complete", {
                'status': 'finished',
                'session_id': request.session_id,
                'timestamp': datetime.now().isoformat(),
                'tools_used': tools_executed,
                'artifacts_count': len(artifacts)
            })

        except Exception as e:
            logger.error(f"Stream processing error: {e}", exc_info=True)
            yield self._create_stream_event("error", {'error': str(e)})

    async def _execute_tool_safe(self, tool, args: dict, tool_name: str) -> Any:
        """OPTIMIZED: Safe tool execution with timeout"""
        try:
            # OPTIMIZED: Add timeout to tool execution
            result = await asyncio.wait_for(tool.arun(args), timeout=15)
            logger.debug(f"Tool {tool_name} executed successfully")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} timed out")
            return f"Tool {tool_name} timed out"
        except Exception as e:
            logger.error(f"Tool {tool_name} execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _extract_email_artifacts(self, result: Any) -> List[Dict[str, str]]:
        """OPTIMIZED: Email artifact extraction with required fields only"""
        artifacts = []

        try:
            # Handle different result formats
            if isinstance(result, str):
                try:
                    email_data = json.loads(result)
                except json.JSONDecodeError:
                    # Try to extract JSON from string if it's wrapped
                    json_match = re.search(r'(\{.*\}|\[.*\])', result, re.DOTALL)
                    if json_match:
                        email_data = json.loads(json_match.group())
                    else:
                        return []
            else:
                email_data = result

            # Extract emails from different possible structures
            emails_list = []

            if isinstance(email_data, list):
                emails_list = email_data
            elif isinstance(email_data, dict):
                # Check common nested structures
                emails_list = (
                        email_data.get("emails", []) or
                        email_data.get("data", []) or
                        email_data.get("result", []) or
                        email_data.get("messages", []) or
                        [email_data]  # Single email
                )

            # OPTIMIZED: Process only first 20 emails for speed
            for i, email in enumerate(emails_list[:20]):
                if not isinstance(email, dict):
                    continue

                artifact = self._create_email_artifact_fast(email, i)
                if artifact:
                    artifacts.append(artifact)

            return artifacts

        except Exception as e:
            logger.error(f"Email artifact extraction error: {e}")
            return []

    def _create_email_artifact_fast(self, email: dict, index: int) -> Dict[str, str]:
        """OPTIMIZED: Faster email artifact creation with required fields only"""
        try:
            # Extract only required fields: sender, subject, date, message_id
            return {
                "sender": str(email.get("from") or email.get("sender") or email.get("From") or "Unknown"),
                "subject": str(email.get("subject") or email.get("Subject") or email.get("title") or "No Subject"),
                "date": str(
                    email.get("date") or email.get("Date") or email.get("timestamp") or datetime.now().isoformat()),
                "message_id": str(email.get("_id") or email.get("id") or email.get(
                    "messageId") or f"msg_{index}_{int(datetime.now().timestamp())}")
            }
        except Exception as e:
            logger.error(f"Error creating artifact from email {index}: {e}")
            return None

    def _create_stream_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Create SSE event"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # OPTIMIZED: Non-streaming chat for simple requests
    @traceable(name="process_chat_request")
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """OPTIMIZED: Fast non-streaming chat for simple requests"""
        try:
            # Skip memory for very simple requests
            memory_context = ""
            if request.use_memory and len(request.message.split()) > 3:
                try:
                    memory_context = await asyncio.wait_for(
                        self.memory_manager.get_session_memory_context(
                            request.session_id, request.message, self.llm
                        ),
                        timeout=1  # Very short timeout
                    )
                except:
                    pass

            system_prompt = self._get_system_prompt(memory_context, request.session_id)
            llm_with_tools = self.llm.bind_tools(self.tools)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]

            # OPTIMIZED: Single LLM call with concurrent tool execution
            response = await llm_with_tools.ainvoke(messages)
            response_content = response.content
            tools_used = []

            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_tasks = []
                for tc in response.tool_calls:
                    tool_name = tc['name']
                    tools_used.append(tool_name)

                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        task = asyncio.create_task(
                            self._execute_tool_safe(tool, tc.get('args', {}), tool_name)
                        )
                        tool_tasks.append(task)

                # Wait for all tools to complete
                if tool_tasks:
                    await asyncio.gather(*tool_tasks, return_exceptions=True)

            # Background memory storage
            if request.use_memory:
                asyncio.create_task(
                    self.memory_manager.add_conversation_to_memory(
                        request.session_id, request.message, response_content
                    )
                )

            return ChatResponse(
                response=response_content,
                session_id=request.session_id,
                tools_used=tools_used,
                memory_used={}
            )

        except Exception as e:
            logger.error(f"Chat request error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Keep other methods unchanged but add this optimization
    async def _discover_and_setup_tools(self):
        """OPTIMIZED: Faster tool rediscovery"""
        if not self.mcp_client:
            return

        try:
            all_tools_data = await asyncio.wait_for(
                self.mcp_client.discover_all_tools(),
                timeout=8  # Reduced timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Tool rediscovery timed out")
            return

        # Clear caches
        self.tools = []
        self._email_tools_cache.clear()

        # Rebuild tools and cache
        for tool_name, tool_data in all_tools_data.items():
            try:
                lc_tool = create_langchain_tool_from_universal_mcp(tool_data, self.mcp_client)
                self.tools.append(lc_tool)

                if any(keyword in tool_name.lower() for keyword in
                       ['email', 'get_emails', 'search_email', 'find_email']):
                    self._email_tools_cache.add(tool_name)

            except Exception as e:
                logger.error(f"Error loading tool {tool_name}: {e}")

        logger.info(f"Rediscovered {len(self.tools)} tools, {len(self._email_tools_cache)} email tools")

    @traceable(name="call_tool_directly")
    async def call_tool_directly(self, request: ToolCallRequest) -> ToolCallResponse:
        """OPTIMIZED: Direct tool calling"""
        try:
            if not self.mcp_client:
                raise HTTPException(status_code=503, detail="MCP client not initialized")

            # OPTIMIZED: Add timeout to direct tool calls
            result = await asyncio.wait_for(
                self.mcp_client.call_tool(request.tool_name, request.arguments),
                timeout=30
            )

            return ToolCallResponse(
                result=result,
                tool_name=request.tool_name,
                session_id=request.session_id,
                success=True
            )

        except asyncio.TimeoutError:
            logger.error(f"Tool {request.tool_name} timed out")
            return ToolCallResponse(
                result="Tool execution timed out",
                tool_name=request.tool_name,
                session_id=request.session_id,
                success=False
            )
        except Exception as e:
            logger.error(f"Error calling tool {request.tool_name}: {e}")
            return ToolCallResponse(
                result=str(e),
                tool_name=request.tool_name,
                session_id=request.session_id,
                success=False
            )

    async def add_server(self, request: AddServerRequest) -> Dict[str, str]:
        """Add new MCP server"""
        try:
            if self.mcp_client:
                self.mcp_client.servers[request.name] = request.url
                await self._discover_and_setup_tools()

            return {"status": "success", "message": f"Server {request.name} added successfully"}

        except Exception as e:
            logger.error(f"Error adding server: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_memory_status(self, session_id: str) -> MemoryStatus:
        """Get memory status"""
        try:
            stats = await self.memory_manager.get_memory_stats(session_id)

            return MemoryStatus(
                redis_conversations=stats.get("session_conversations", 0),
                window_size=self.memory_manager.window_size,
                vector_store_status="available" if stats.get("vector_store_available") else "unavailable",
                available_tools=len(self.tools),
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_health_status(self) -> HealthStatus:
        """Get system health"""
        components = {
            "llm": "healthy",
            "memory_manager": "healthy" if self.memory_manager.redis else "degraded",
            "mcp_client": "healthy" if self.mcp_client else "unavailable",
            "vector_store": "healthy" if self.memory_manager.vector_store else "unavailable"
        }

        overall_status = "healthy"
        if "unavailable" in components.values():
            overall_status = "degraded"
        elif "degraded" in components.values():
            overall_status = "degraded"

        uptime = (datetime.now() - self._start_time).total_seconds()

        return HealthStatus(
            status=overall_status,
            components=components,
            uptime=uptime,
            version="1.0.0"
        )

    async def list_available_tools(self) -> List[Dict[str, str]]:
        """List available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "server": getattr(tool, '_server', 'unknown')
            }
            for tool in self.tools
        ]

    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.close()


# Global agent instance
agent_manager = AgentManager()