from datetime import datetime
from .mcp import UniversalMCPClient
from langchain_core.tools import StructuredTool
from typing import Optional, List, Dict, Any
from pydantic import Field, create_model, BaseModel
import uuid
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


class StreamEvent(BaseModel):
    event_type: str = Field(..., description="Type of stream event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthStatus(BaseModel):
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")
    version: str = Field(default="1.0.0", description="API version")
