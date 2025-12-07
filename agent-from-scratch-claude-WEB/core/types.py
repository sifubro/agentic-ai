"""
Core types and base classes for the Agentic Orchestration Framework.
Fully type-annotated with no external framework dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, 
    Optional, Set, Tuple, TypeVar, Union, Protocol,
    runtime_checkable, Awaitable
)
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TaskStatus(Enum):
    """Status of a task in the graph."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    RETRYING = "retrying"


class AgentRole(Enum):
    """Role types for agents in the system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    ARBITRATOR = "arbitrator"


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    NEGOTIATION = "negotiation"
    DEBATE = "debate"
    ARBITRATION = "arbitration"
    STREAMING = "streaming"


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ModelProvider(Enum):
    """AI model providers for fallback chain."""
    GPT4O = "gpt-4o"
    GEMINI = "gemini-pro"
    CLAUDE = "claude-3-opus"
    LOCAL = "local"


# ============================================================================
# BASE DATA CLASSES
# ============================================================================

@dataclass
class JSONSerializable:
    """Base class for JSON-serializable objects."""
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'JSONSerializable':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class Message(JSONSerializable):
    """Message structure for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender_id: str = ""
    receiver_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: int = 3600  # Time to live in seconds
    priority: int = 5  # 1-10, higher is more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum handling."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl,
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('type'), str):
            data['type'] = MessageType(data['type'])
        return cls(**data)


@dataclass
class TaskNode(JSONSerializable):
    """A node in the task graph representing a single task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # List of task IDs
    assigned_agent: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum handling."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskNode':
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get('status'), str):
            data['status'] = TaskStatus(data['status'])
        return cls(**data)


@dataclass
class MemoryEntry(JSONSerializable):
    """An entry in agent memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum handling."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, Enum) else self.memory_type,
            "content": self.content,
            "embedding": self.embedding,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "tags": self.tags
        }


@dataclass
class Session(JSONSerializable):
    """Session information for tracking agent interactions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str = field(default_factory=lambda: (datetime.utcnow() + timedelta(hours=24)).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    messages: List[str] = field(default_factory=list)  # Message IDs
    task_graph_id: Optional[str] = None


@dataclass
class AuthToken(JSONSerializable):
    """Authentication token structure."""
    token: str = ""
    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: str = field(default_factory=lambda: (datetime.utcnow() + timedelta(hours=1)).isoformat())
    refresh_token: Optional[str] = None
    refresh_expires_at: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    is_revoked: bool = False


@dataclass
class User(JSONSerializable):
    """User account information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    password_hash: str = ""
    salt: str = ""
    email: str = ""
    roles: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_login: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TaskGraph(JSONSerializable):
    """A graph of tasks to be executed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (from_id, to_id)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    owner_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "owner_id": self.owner_id,
            "metadata": self.metadata
        }


# ============================================================================
# PROTOCOLS (Interface definitions)
# ============================================================================

@runtime_checkable
class Agent(Protocol):
    """Protocol for agent implementations."""
    id: str
    name: str
    role: AgentRole
    
    async def process_message(self, message: Message) -> Message:
        """Process an incoming message and return a response."""
        ...
    
    async def execute_task(self, task: TaskNode) -> TaskNode:
        """Execute a task and return the result."""
        ...


@runtime_checkable
class MessageQueue(Protocol):
    """Protocol for message queue implementations."""
    
    async def publish(self, queue: str, message: Message) -> bool:
        """Publish a message to a queue."""
        ...
    
    async def subscribe(self, queue: str, callback: Callable[[Message], Awaitable[None]]) -> None:
        """Subscribe to a queue with a callback."""
        ...
    
    async def close(self) -> None:
        """Close the connection."""
        ...


@runtime_checkable
class Storage(Protocol):
    """Protocol for storage implementations."""
    
    async def save(self, collection: str, key: str, data: Dict[str, Any]) -> bool:
        """Save data to storage."""
        ...
    
    async def load(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """Load data from storage."""
        ...
    
    async def delete(self, collection: str, key: str) -> bool:
        """Delete data from storage."""
        ...
    
    async def query(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query data with filters."""
        ...


# ============================================================================
# EXCEPTIONS
# ============================================================================

class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class TaskExecutionError(AgentError):
    """Error during task execution."""
    pass


class GraphCycleError(AgentError):
    """Error when a cycle is detected in the task graph."""
    pass


class AuthenticationError(AgentError):
    """Authentication related errors."""
    pass


class AuthorizationError(AgentError):
    """Authorization related errors."""
    pass


class MessageParsingError(AgentError):
    """Error parsing messages."""
    pass


class StorageError(AgentError):
    """Storage related errors."""
    pass


class TimeoutError(AgentError):
    """Timeout errors."""
    pass


class ModelFallbackError(AgentError):
    """Error when all model fallbacks fail."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def generate_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def current_timestamp() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat()


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def is_expired(expires_at: str) -> bool:
    """Check if a timestamp has expired."""
    return datetime.utcnow() > parse_timestamp(expires_at)