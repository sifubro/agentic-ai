"""
Message protocol for inter-agent communication.
All messages are JSON-serializable for network transport.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MessageType(Enum):
    """Types of messages in the system."""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    HANDOFF = "handoff"
    QUERY = "query"
    RESPONSE = "response"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    NEGOTIATION = "negotiation"
    ARBITRATION = "arbitration"
    MEMORY_RECALL = "memory_recall"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"


class MessagePriority(Enum):
    """Priority levels for message processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """
    JSON-serializable message for inter-agent communication.
    """
    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
        return json.dumps(data, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return json.loads(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> Message:
        """Deserialize message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create message from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            priority=MessagePriority(data.get("priority", 1)),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            ttl=data.get("ttl"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )

    def create_reply(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        **kwargs: Any
    ) -> Message:
        """Create a reply message to this message."""
        return Message(
            message_type=message_type,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            payload=payload,
            correlation_id=self.message_id,
            session_id=self.session_id,
            **kwargs
        )

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        elapsed = (datetime.utcnow() - created).total_seconds()
        return elapsed > self.ttl

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> Message:
        """Create a new message with incremented retry count."""
        return Message(
            message_id=self.message_id,
            message_type=self.message_type,
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            payload=self.payload,
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
            priority=self.priority,
            metadata=self.metadata,
            session_id=self.session_id,
            ttl=self.ttl,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
        )


@dataclass
class MessageBatch:
    """A batch of messages for efficient processing."""
    messages: List[Message]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        """Serialize batch to JSON."""
        return json.dumps({
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "messages": [msg.to_dict() for msg in self.messages]
        })

    @classmethod
    def from_json(cls, json_str: str) -> MessageBatch:
        """Deserialize batch from JSON."""
        data = json.loads(json_str)
        return cls(
            batch_id=data["batch_id"],
            created_at=data["created_at"],
            messages=[Message.from_dict(m) for m in data["messages"]]
        )


class MessageQueue:
    """Thread-safe message queue with priority support."""

    def __init__(self, max_size: int = 10000):
        self._queue: List[Message] = []
        self._max_size = max_size
        self._lock = None  # Will be set in async context

    def push(self, message: Message) -> bool:
        """Add message to queue with priority ordering."""
        if len(self._queue) >= self._max_size:
            return False

        # Insert based on priority (higher priority first)
        insert_idx = 0
        for i, msg in enumerate(self._queue):
            if message.priority.value > msg.priority.value:
                insert_idx = i
                break
            insert_idx = i + 1

        self._queue.insert(insert_idx, message)
        return True

    def pop(self) -> Optional[Message]:
        """Remove and return highest priority message."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def peek(self) -> Optional[Message]:
        """View highest priority message without removing."""
        if not self._queue:
            return None
        return self._queue[0]

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def clear(self) -> None:
        """Clear all messages from queue."""
        self._queue.clear()

    def get_by_correlation_id(self, correlation_id: str) -> List[Message]:
        """Get all messages with a specific correlation ID."""
        return [m for m in self._queue if m.correlation_id == correlation_id]
