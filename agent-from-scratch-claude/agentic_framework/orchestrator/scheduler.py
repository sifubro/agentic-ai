"""
Task scheduler with multiple scheduling strategies.
"""

from __future__ import annotations
import asyncio
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.graph import TaskGraph, TaskNode, NodeStatus


class SchedulingStrategy(Enum):
    """Available scheduling strategies."""
    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_FIRST = "shortest_first"
    CRITICAL_PATH = "critical_path"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"


@dataclass(order=True)
class ScheduledTask:
    """A task scheduled for execution."""
    priority: int
    scheduled_at: str = field(compare=False)
    node_id: str = field(compare=False)
    assigned_agent: Optional[str] = field(compare=False, default=None)
    deadline: Optional[str] = field(compare=False, default=None)
    estimated_duration: float = field(compare=False, default=0.0)


class BaseScheduler(ABC):
    """Abstract base scheduler."""

    @abstractmethod
    async def schedule(
        self,
        graph: TaskGraph,
        available_agents: List[str]
    ) -> List[ScheduledTask]:
        """Schedule tasks from graph to agents."""
        pass

    @abstractmethod
    async def reschedule(
        self,
        graph: TaskGraph,
        failed_node_id: str,
        available_agents: List[str]
    ) -> Optional[ScheduledTask]:
        """Reschedule a failed task."""
        pass


class Scheduler(BaseScheduler):
    """
    Task scheduler supporting multiple scheduling strategies.
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY,
        max_concurrent_per_agent: int = 5,
    ):
        self.strategy = strategy
        self.max_concurrent_per_agent = max_concurrent_per_agent
        self._agent_loads: Dict[str, int] = {}
        self._agent_round_robin_idx = 0
        self._scheduled_tasks: List[ScheduledTask] = []
        self._task_history: List[Dict[str, Any]] = []

    async def schedule(
        self,
        graph: TaskGraph,
        available_agents: List[str]
    ) -> List[ScheduledTask]:
        """
        Schedule ready tasks to available agents.
        """
        ready_nodes = graph.get_ready_nodes()
        if not ready_nodes or not available_agents:
            return []

        # Initialize agent loads
        for agent in available_agents:
            if agent not in self._agent_loads:
                self._agent_loads[agent] = 0

        # Order nodes based on strategy
        ordered_nodes = await self._order_nodes(ready_nodes, graph)

        scheduled = []
        for node in ordered_nodes:
            agent = await self._select_agent(node, available_agents)
            if agent:
                task = ScheduledTask(
                    priority=-node.priority,  # Negative for min-heap
                    scheduled_at=datetime.utcnow().isoformat(),
                    node_id=node.node_id,
                    assigned_agent=agent,
                    estimated_duration=node.timeout,
                )
                scheduled.append(task)
                node.assigned_agent = agent
                self._agent_loads[agent] += 1
                self._scheduled_tasks.append(task)

        return scheduled

    async def _order_nodes(
        self,
        nodes: List[TaskNode],
        graph: TaskGraph
    ) -> List[TaskNode]:
        """Order nodes based on scheduling strategy."""
        if self.strategy == SchedulingStrategy.FIFO:
            return sorted(nodes, key=lambda n: n.created_at)

        elif self.strategy == SchedulingStrategy.PRIORITY:
            return sorted(nodes, key=lambda n: -n.priority)

        elif self.strategy == SchedulingStrategy.SHORTEST_FIRST:
            return sorted(nodes, key=lambda n: n.timeout)

        elif self.strategy == SchedulingStrategy.CRITICAL_PATH:
            critical_path = set(graph.get_critical_path())
            return sorted(
                nodes,
                key=lambda n: (n.node_id not in critical_path, -n.priority)
            )

        else:  # Default to priority
            return sorted(nodes, key=lambda n: -n.priority)

    async def _select_agent(
        self,
        node: TaskNode,
        available_agents: List[str]
    ) -> Optional[str]:
        """Select an agent for the task."""
        # Filter agents at capacity
        eligible = [
            a for a in available_agents
            if self._agent_loads.get(a, 0) < self.max_concurrent_per_agent
        ]

        if not eligible:
            return None

        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            agent = eligible[self._agent_round_robin_idx % len(eligible)]
            self._agent_round_robin_idx += 1
            return agent

        elif self.strategy == SchedulingStrategy.LOAD_BALANCED:
            # Select agent with lowest load
            return min(eligible, key=lambda a: self._agent_loads.get(a, 0))

        else:
            # Default: first available
            return eligible[0] if eligible else None

    async def reschedule(
        self,
        graph: TaskGraph,
        failed_node_id: str,
        available_agents: List[str]
    ) -> Optional[ScheduledTask]:
        """Reschedule a failed task to a different agent."""
        try:
            node = graph.get_node(failed_node_id)
        except Exception:
            return None

        if not node.can_retry():
            return None

        # Try to assign to a different agent
        previous_agent = node.assigned_agent
        eligible = [a for a in available_agents if a != previous_agent]

        if eligible:
            agent = await self._select_agent(node, eligible)
        else:
            agent = await self._select_agent(node, available_agents)

        if agent:
            node.status = NodeStatus.READY
            task = ScheduledTask(
                priority=-node.priority,
                scheduled_at=datetime.utcnow().isoformat(),
                node_id=node.node_id,
                assigned_agent=agent,
                estimated_duration=node.timeout,
            )
            node.assigned_agent = agent
            self._agent_loads[agent] = self._agent_loads.get(agent, 0) + 1
            return task

        return None

    def mark_completed(self, node_id: str, agent_id: str) -> None:
        """Mark a task as completed and update agent load."""
        if agent_id in self._agent_loads:
            self._agent_loads[agent_id] = max(0, self._agent_loads[agent_id] - 1)

        self._task_history.append({
            "node_id": node_id,
            "agent_id": agent_id,
            "completed_at": datetime.utcnow().isoformat(),
        })

    def mark_failed(self, node_id: str, agent_id: str) -> None:
        """Mark a task as failed and update agent load."""
        if agent_id in self._agent_loads:
            self._agent_loads[agent_id] = max(0, self._agent_loads[agent_id] - 1)

    def get_agent_load(self, agent_id: str) -> int:
        """Get current load for an agent."""
        return self._agent_loads.get(agent_id, 0)

    def get_all_loads(self) -> Dict[str, int]:
        """Get loads for all agents."""
        return self._agent_loads.copy()

    def reset_loads(self) -> None:
        """Reset all agent loads."""
        self._agent_loads.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        return {
            "total_scheduled": len(self._scheduled_tasks),
            "total_completed": len(self._task_history),
            "agent_loads": self._agent_loads.copy(),
            "strategy": self.strategy.value,
        }


class AdaptiveScheduler(Scheduler):
    """
    Adaptive scheduler that adjusts strategy based on performance metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._performance_history: List[Dict[str, Any]] = []
        self._strategy_scores: Dict[SchedulingStrategy, float] = {
            s: 1.0 for s in SchedulingStrategy
        }

    async def schedule(
        self,
        graph: TaskGraph,
        available_agents: List[str]
    ) -> List[ScheduledTask]:
        """Schedule with adaptive strategy selection."""
        # Periodically adjust strategy based on performance
        if len(self._performance_history) % 10 == 0 and self._performance_history:
            await self._adapt_strategy()

        return await super().schedule(graph, available_agents)

    async def _adapt_strategy(self) -> None:
        """Adapt scheduling strategy based on performance."""
        if not self._performance_history:
            return

        # Calculate average completion time per strategy
        strategy_times: Dict[SchedulingStrategy, List[float]] = {
            s: [] for s in SchedulingStrategy
        }

        for record in self._performance_history[-100:]:
            strategy = SchedulingStrategy(record.get("strategy", "priority"))
            completion_time = record.get("completion_time", 0)
            strategy_times[strategy].append(completion_time)

        # Update scores
        for strategy, times in strategy_times.items():
            if times:
                avg_time = sum(times) / len(times)
                # Lower time is better, so invert
                self._strategy_scores[strategy] = 1.0 / (avg_time + 0.1)

        # Select best strategy
        best_strategy = max(
            self._strategy_scores.keys(),
            key=lambda s: self._strategy_scores[s]
        )
        self.strategy = best_strategy

    def record_completion(
        self,
        node_id: str,
        completion_time: float
    ) -> None:
        """Record task completion for adaptation."""
        self._performance_history.append({
            "node_id": node_id,
            "completion_time": completion_time,
            "strategy": self.strategy.value,
            "timestamp": datetime.utcnow().isoformat(),
        })
