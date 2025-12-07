"""
Agentic Orchestration Framework
A complete graph-based multi-agent orchestration system built from scratch.
"""

__version__ = "1.0.0"
__author__ = "Agentic Framework"

from .core.agent import Agent, AgentConfig, AgentRole
from .core.message import Message, MessageType
from .core.graph import TaskGraph, TaskNode, NodeStatus
from .orchestrator.orchestrator import Orchestrator
from .auth.auth_manager import AuthManager
from .memory.memory_manager import MemoryManager
from .storage.sqlite_storage import SQLiteStorage

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentRole",
    "Message",
    "MessageType",
    "TaskGraph",
    "TaskNode",
    "NodeStatus",
    "Orchestrator",
    "AuthManager",
    "MemoryManager",
    "SQLiteStorage",
]
