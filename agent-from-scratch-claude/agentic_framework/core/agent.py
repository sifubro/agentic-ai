"""
Base Agent classes and configuration.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from .message import Message, MessageType, MessageQueue, MessagePriority

if TYPE_CHECKING:
    from ..memory.memory_manager import MemoryManager


class AgentRole(Enum):
    """Predefined agent roles in the system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    TOOL_USER = "tool_user"
    MEMORY_KEEPER = "memory_keeper"
    ARBITRATOR = "arbitrator"


class AgentStatus(Enum):
    """Agent lifecycle status."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: AgentRole
    model_provider: str = "local"  # local, openai, anthropic, google
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    timeout: float = 300.0  # seconds
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2.0,
        "initial_delay": 1.0
    })
    fallback_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout": self.timeout,
            "retry_policy": self.retry_policy,
            "fallback_models": self.fallback_models,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """Create config from dictionary."""
        data = data.copy()
        data["role"] = AgentRole(data["role"])
        return cls(**data)


@dataclass
class AgentMetrics:
    """Metrics for agent performance monitoring."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    last_activity: Optional[str] = None

    def record_task_completion(self, duration: float) -> None:
        """Record a completed task."""
        self.tasks_completed += 1
        self.total_processing_time += duration
        self.last_activity = datetime.utcnow().isoformat()

    def record_task_failure(self, error: str) -> None:
        """Record a failed task."""
        self.tasks_failed += 1
        self.errors.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        })
        self.last_activity = datetime.utcnow().isoformat()

    @property
    def average_processing_time(self) -> float:
        """Calculate average task processing time."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_completed

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total


class Agent(ABC):
    """
    Base class for all agents in the system.
    """

    def __init__(
        self,
        config: AgentConfig,
        agent_id: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.inbox = MessageQueue()
        self.outbox = MessageQueue()
        self.memory_manager = memory_manager
        self._running = False
        self._current_tasks: Set[str] = set()
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default message handlers."""
        self._message_handlers = {
            MessageType.TASK: self._handle_task,
            MessageType.QUERY: self._handle_query,
            MessageType.CONTROL: self._handle_control,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.NEGOTIATION: self._handle_negotiation,
        }

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any]
    ) -> None:
        """Register a custom message handler."""
        self._message_handlers[message_type] = handler

    async def start(self) -> None:
        """Start the agent's message processing loop."""
        self._running = True
        self.status = AgentStatus.IDLE
        await self._initialize()
        asyncio.create_task(self._message_loop())

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
        self.status = AgentStatus.SHUTDOWN
        await self._cleanup()

    async def _initialize(self) -> None:
        """Initialize agent resources. Override in subclasses."""
        pass

    async def _cleanup(self) -> None:
        """Cleanup agent resources. Override in subclasses."""
        pass

    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                message = self.inbox.pop()
                if message:
                    await self._process_message(message)
                else:
                    await asyncio.sleep(0.01)  # Prevent busy waiting
            except Exception as e:
                self.metrics.record_task_failure(str(e))
                self.status = AgentStatus.ERROR

    async def _process_message(self, message: Message) -> None:
        """Process an incoming message."""
        self.metrics.messages_received += 1

        if message.is_expired():
            return

        handler = self._message_handlers.get(message.message_type)
        if handler:
            self.status = AgentStatus.BUSY
            start_time = asyncio.get_event_loop().time()

            try:
                result = await handler(message)
                duration = asyncio.get_event_loop().time() - start_time
                self.metrics.record_task_completion(duration)

                if result:
                    await self.send_message(result)
            except Exception as e:
                self.metrics.record_task_failure(str(e))
                error_response = message.create_reply(
                    MessageType.ERROR,
                    {"error": str(e), "agent_id": self.agent_id}
                )
                await self.send_message(error_response)
            finally:
                self.status = AgentStatus.IDLE

    async def send_message(self, message: Message) -> None:
        """Send a message to another agent."""
        self.outbox.push(message)
        self.metrics.messages_sent += 1

    async def receive_message(self, message: Message) -> None:
        """Receive a message from another agent."""
        self.inbox.push(message)

    @abstractmethod
    async def _handle_task(self, message: Message) -> Optional[Message]:
        """Handle a task message. Must be implemented by subclasses."""
        pass

    async def _handle_query(self, message: Message) -> Optional[Message]:
        """Handle a query message."""
        return message.create_reply(
            MessageType.RESPONSE,
            {
                "agent_id": self.agent_id,
                "status": self.status.value,
                "metrics": {
                    "tasks_completed": self.metrics.tasks_completed,
                    "tasks_failed": self.metrics.tasks_failed,
                    "success_rate": self.metrics.success_rate,
                }
            }
        )

    async def _handle_control(self, message: Message) -> Optional[Message]:
        """Handle a control message."""
        command = message.payload.get("command")
        if command == "stop":
            await self.stop()
        elif command == "status":
            return message.create_reply(
                MessageType.STATUS,
                {"status": self.status.value, "agent_id": self.agent_id}
            )
        return None

    async def _handle_heartbeat(self, message: Message) -> Optional[Message]:
        """Handle a heartbeat message."""
        return message.create_reply(
            MessageType.HEARTBEAT,
            {"status": "alive", "agent_id": self.agent_id}
        )

    async def _handle_negotiation(self, message: Message) -> Optional[Message]:
        """Handle negotiation messages for multi-agent collaboration."""
        proposal = message.payload.get("proposal")
        # Default implementation accepts all proposals
        return message.create_reply(
            MessageType.NEGOTIATION,
            {"response": "accept", "proposal_id": proposal.get("id") if proposal else None}
        )

    async def recall_memory(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories."""
        if self.memory_manager:
            return await self.memory_manager.recall(
                agent_id=self.agent_id,
                query=query,
                limit=limit
            )
        return []

    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new memory."""
        if self.memory_manager:
            return await self.memory_manager.store(
                agent_id=self.agent_id,
                content=content,
                memory_type=memory_type,
                metadata=metadata or {}
            )
        return ""

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "role": self.config.role.value,
            "status": self.status.value,
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": self.metrics.success_rate,
                "average_processing_time": self.metrics.average_processing_time,
            },
            "capabilities": self.config.capabilities,
        }


class LocalAgent(Agent):
    """Agent that runs tasks locally without external API calls."""

    async def _handle_task(self, message: Message) -> Optional[Message]:
        """Execute a task locally."""
        task = message.payload.get("task")
        task_type = message.payload.get("type", "default")

        result = await self._execute_local_task(task, task_type)

        return message.create_reply(
            MessageType.RESULT,
            {"result": result, "agent_id": self.agent_id}
        )

    async def _execute_local_task(
        self,
        task: Any,
        task_type: str
    ) -> Dict[str, Any]:
        """Execute a local task. Override for custom logic."""
        return {
            "status": "completed",
            "task": task,
            "type": task_type,
            "processed_by": self.agent_id
        }


class LLMAgent(Agent):
    """Agent that uses LLM APIs for task execution."""

    def __init__(
        self,
        config: AgentConfig,
        api_client: Optional[Any] = None,
        **kwargs: Any
    ):
        super().__init__(config, **kwargs)
        self.api_client = api_client
        self._current_model_idx = 0

    async def _handle_task(self, message: Message) -> Optional[Message]:
        """Execute a task using LLM."""
        task = message.payload.get("task")
        context = message.payload.get("context", {})

        # Try to recall relevant memories
        memories = await self.recall_memory(str(task))

        try:
            result = await self._call_llm(task, context, memories)
            return message.create_reply(
                MessageType.RESULT,
                {"result": result, "agent_id": self.agent_id}
            )
        except Exception as e:
            # Try fallback models
            result = await self._try_fallback_models(task, context, memories)
            if result:
                return message.create_reply(
                    MessageType.RESULT,
                    {"result": result, "agent_id": self.agent_id, "fallback": True}
                )
            raise

    async def _call_llm(
        self,
        task: Any,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call the LLM API. Override for specific implementations."""
        # Base implementation - should be overridden
        return {
            "response": f"Processed task: {task}",
            "model": self.config.model_name,
            "context_used": bool(context),
            "memories_used": len(memories)
        }

    async def _try_fallback_models(
        self,
        task: Any,
        context: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Try fallback models on failure."""
        for model in self.config.fallback_models:
            try:
                original_model = self.config.model_name
                self.config.model_name = model
                result = await self._call_llm(task, context, memories)
                self.config.model_name = original_model
                return result
            except Exception:
                continue
        return None
