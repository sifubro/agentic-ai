"""
Main orchestrator for the agentic framework.
Coordinates agents, executes task graphs, and manages the overall workflow.
"""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from ..core.graph import TaskGraph, TaskNode, NodeStatus, CycleDetectedError
from ..core.message import Message, MessageType, MessageQueue
from ..core.agent import Agent, AgentConfig, AgentStatus
from .executor import TaskExecutor, ExecutionResult, ExecutionMode
from .scheduler import Scheduler, SchedulingStrategy

if TYPE_CHECKING:
    from ..memory.memory_manager import MemoryManager
    from ..storage.sqlite_storage import SQLiteStorage


logger = logging.getLogger(__name__)


class OrchestratorStatus(Enum):
    """Status of the orchestrator."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_concurrent_graphs: int = 10
    max_retries: int = 3
    default_timeout: float = 300.0
    enable_self_healing: bool = True
    enable_persistence: bool = True
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    checkpoint_interval: int = 10  # Save state every N completed tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphExecution:
    """Tracks execution of a task graph."""
    graph_id: str
    graph: TaskGraph
    status: OrchestratorStatus = OrchestratorStatus.IDLE
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """
    Main orchestrator that coordinates agents and executes task graphs.
    Supports self-healing, checkpointing, and dynamic graph mutation.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        memory_manager: Optional[MemoryManager] = None,
        storage: Optional[SQLiteStorage] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.memory_manager = memory_manager
        self.storage = storage

        self._status = OrchestratorStatus.IDLE
        self._agents: Dict[str, Agent] = {}
        self._graphs: Dict[str, GraphExecution] = {}
        self._executor: Optional[TaskExecutor] = None
        self._scheduler: Optional[Scheduler] = None
        self._message_router = MessageQueue()
        self._running = False
        self._checkpoint_counter = 0

        self._event_handlers: Dict[str, List[Callable]] = {
            "graph_started": [],
            "graph_completed": [],
            "node_completed": [],
            "node_failed": [],
            "error": [],
            "checkpoint": [],
        }

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        self._executor = TaskExecutor(
            default_mode=self.config.execution_mode,
            default_timeout=self.config.default_timeout,
        )
        await self._executor.initialize()

        self._scheduler = Scheduler(
            strategy=self.config.scheduling_strategy,
        )

        self._running = True
        self._status = OrchestratorStatus.IDLE

        # Start background tasks
        asyncio.create_task(self._message_routing_loop())
        asyncio.create_task(self._health_check_loop())

        if self.config.enable_persistence and self.storage:
            await self._restore_state()

        logger.info("Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        self._running = False
        self._status = OrchestratorStatus.SHUTDOWN

        # Save state before shutdown
        if self.config.enable_persistence and self.storage:
            await self._save_state()

        # Stop all agents
        for agent in self._agents.values():
            await agent.stop()

        # Shutdown executor
        if self._executor:
            await self._executor.shutdown()

        logger.info("Orchestrator shutdown complete")

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.agent_id] = agent
        if self._executor:
            self._executor.register_agent(agent)
        logger.info(f"Agent registered: {agent.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            if self._executor:
                self._executor.unregister_agent(agent_id)
            logger.info(f"Agent unregistered: {agent_id}")

    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, **kwargs: Any) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(**kwargs)
                else:
                    handler(**kwargs)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def submit_graph(
        self,
        graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None,
        start_immediately: bool = True
    ) -> str:
        """
        Submit a task graph for execution.
        Returns graph_id for tracking.
        """
        # Validate graph
        try:
            graph.topological_sort()
        except CycleDetectedError as e:
            raise ValueError(f"Invalid graph: {e}")

        # Check concurrent limit
        running_count = sum(
            1 for g in self._graphs.values()
            if g.status == OrchestratorStatus.RUNNING
        )
        if running_count >= self.config.max_concurrent_graphs:
            raise RuntimeError("Maximum concurrent graphs reached")

        # Create execution tracking
        execution = GraphExecution(
            graph_id=graph.graph_id,
            graph=graph,
            context=context or {},
        )
        self._graphs[graph.graph_id] = execution

        # Assign agents to nodes if not assigned
        await self._assign_agents_to_graph(graph)

        if start_immediately:
            asyncio.create_task(self._execute_graph(graph.graph_id))

        logger.info(f"Graph submitted: {graph.graph_id}")
        return graph.graph_id

    async def _assign_agents_to_graph(self, graph: TaskGraph) -> None:
        """Assign agents to unassigned nodes in graph."""
        available_agents = list(self._agents.keys())
        if not available_agents:
            logger.warning("No agents available for assignment")
            return

        for node in graph.get_nodes():
            if not node.assigned_agent:
                # Simple round-robin assignment
                # Could be enhanced with capability matching
                idx = hash(node.node_id) % len(available_agents)
                node.assigned_agent = available_agents[idx]

    async def _execute_graph(self, graph_id: str) -> None:
        """Execute a task graph."""
        execution = self._graphs.get(graph_id)
        if not execution:
            return

        execution.status = OrchestratorStatus.RUNNING
        execution.started_at = datetime.utcnow().isoformat()
        self._status = OrchestratorStatus.RUNNING

        await self._emit_event("graph_started", graph_id=graph_id)

        try:
            while self._running:
                # Get ready nodes
                ready_nodes = execution.graph.get_ready_nodes()

                if not ready_nodes:
                    # Check if all nodes are complete
                    all_complete = all(
                        n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)
                        for n in execution.graph.get_nodes()
                    )
                    if all_complete:
                        break
                    await asyncio.sleep(0.1)
                    continue

                # Schedule and execute ready nodes in parallel
                available_agents = [
                    a.agent_id for a in self._agents.values()
                    if a.status != AgentStatus.SHUTDOWN
                ]

                scheduled = await self._scheduler.schedule(
                    execution.graph,
                    available_agents
                )

                if not scheduled:
                    await asyncio.sleep(0.1)
                    continue

                # Execute scheduled tasks
                nodes_to_execute = [
                    execution.graph.get_node(s.node_id)
                    for s in scheduled
                ]

                results = await self._executor.execute_parallel(
                    nodes_to_execute,
                    execution.context
                )

                # Process results
                for result in results:
                    execution.results[result.node_id] = result

                    if result.success:
                        await self._handle_node_success(execution, result)
                    else:
                        await self._handle_node_failure(execution, result)

                    # Checkpoint if needed
                    self._checkpoint_counter += 1
                    if self._checkpoint_counter >= self.config.checkpoint_interval:
                        await self._checkpoint(execution)
                        self._checkpoint_counter = 0

            # Mark graph complete
            execution.status = OrchestratorStatus.IDLE
            execution.completed_at = datetime.utcnow().isoformat()

            await self._emit_event(
                "graph_completed",
                graph_id=graph_id,
                results=execution.results
            )

        except Exception as e:
            execution.status = OrchestratorStatus.ERROR
            execution.errors.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            await self._emit_event("error", graph_id=graph_id, error=str(e))
            logger.error(f"Graph execution error: {e}")

        finally:
            # Update orchestrator status
            running_graphs = sum(
                1 for g in self._graphs.values()
                if g.status == OrchestratorStatus.RUNNING
            )
            if running_graphs == 0:
                self._status = OrchestratorStatus.IDLE

    async def _handle_node_success(
        self,
        execution: GraphExecution,
        result: ExecutionResult
    ) -> None:
        """Handle successful node execution."""
        node = execution.graph.get_node(result.node_id)

        # Update context with result
        if result.result:
            execution.context[f"result_{result.node_id}"] = result.result

        # Update scheduler
        self._scheduler.mark_completed(result.node_id, result.agent_id or "")

        await self._emit_event(
            "node_completed",
            graph_id=execution.graph_id,
            node_id=result.node_id,
            result=result
        )

        logger.info(f"Node completed: {result.node_id}")

    async def _handle_node_failure(
        self,
        execution: GraphExecution,
        result: ExecutionResult
    ) -> None:
        """Handle failed node execution with self-healing."""
        node = execution.graph.get_node(result.node_id)

        execution.errors.append({
            "node_id": result.node_id,
            "error": result.error,
            "timestamp": datetime.utcnow().isoformat()
        })

        await self._emit_event(
            "node_failed",
            graph_id=execution.graph_id,
            node_id=result.node_id,
            error=result.error
        )

        # Self-healing: try to reschedule
        if self.config.enable_self_healing and node.status == NodeStatus.RETRYING:
            available_agents = list(self._agents.keys())
            rescheduled = await self._scheduler.reschedule(
                execution.graph,
                result.node_id,
                available_agents
            )
            if rescheduled:
                logger.info(f"Node rescheduled: {result.node_id} -> {rescheduled.assigned_agent}")
                return

        # Mark scheduler
        self._scheduler.mark_failed(result.node_id, result.agent_id or "")

        # Skip dependent nodes
        await self._skip_dependents(execution, result.node_id)

        logger.warning(f"Node failed: {result.node_id} - {result.error}")

    async def _skip_dependents(
        self,
        execution: GraphExecution,
        failed_node_id: str
    ) -> None:
        """Skip all nodes that depend on a failed node."""
        to_skip = set()
        queue = [failed_node_id]

        while queue:
            node_id = queue.pop(0)
            children = execution.graph.get_children(node_id)
            for child in children:
                if child.node_id not in to_skip:
                    to_skip.add(child.node_id)
                    queue.append(child.node_id)

        for node_id in to_skip:
            node = execution.graph.get_node(node_id)
            node.status = NodeStatus.SKIPPED
            logger.info(f"Node skipped due to dependency failure: {node_id}")

    async def _checkpoint(self, execution: GraphExecution) -> None:
        """Save checkpoint of execution state."""
        if self.storage:
            await self.storage.save_graph(
                execution.graph_id,
                execution.graph.to_json()
            )
            await self._emit_event("checkpoint", graph_id=execution.graph_id)
            logger.debug(f"Checkpoint saved: {execution.graph_id}")

    async def _save_state(self) -> None:
        """Save orchestrator state to storage."""
        if not self.storage:
            return

        for execution in self._graphs.values():
            await self.storage.save_graph(
                execution.graph_id,
                execution.graph.to_json()
            )

    async def _restore_state(self) -> None:
        """Restore orchestrator state from storage."""
        if not self.storage:
            return

        graphs = await self.storage.load_all_graphs()
        for graph_id, graph_json in graphs.items():
            try:
                graph = TaskGraph.from_json(graph_json)
                execution = GraphExecution(
                    graph_id=graph_id,
                    graph=graph,
                )
                self._graphs[graph_id] = execution
                logger.info(f"Restored graph: {graph_id}")
            except Exception as e:
                logger.error(f"Failed to restore graph {graph_id}: {e}")

    async def _message_routing_loop(self) -> None:
        """Route messages between agents."""
        while self._running:
            # Collect messages from all agent outboxes
            for agent in self._agents.values():
                while not agent.outbox.is_empty():
                    message = agent.outbox.pop()
                    if message:
                        await self._route_message(message)

            await asyncio.sleep(0.01)

    async def _route_message(self, message: Message) -> None:
        """Route a message to its destination."""
        receiver = self._agents.get(message.receiver_id)
        if receiver:
            await receiver.receive_message(message)
        else:
            logger.warning(f"Unknown receiver: {message.receiver_id}")

    async def _health_check_loop(self) -> None:
        """Periodically check health of agents."""
        while self._running:
            for agent in self._agents.values():
                if agent.status == AgentStatus.ERROR:
                    logger.warning(f"Agent {agent.agent_id} in error state")
                    # Could trigger recovery here

            await asyncio.sleep(30)  # Check every 30 seconds

    # Dynamic Graph Mutation Methods

    async def add_node_to_graph(
        self,
        graph_id: str,
        node: TaskNode,
        dependencies: Optional[List[str]] = None,
        dependents: Optional[List[str]] = None
    ) -> None:
        """Add a node to a running graph."""
        execution = self._graphs.get(graph_id)
        if not execution:
            raise ValueError(f"Graph {graph_id} not found")

        execution.graph.add_node(node)

        # Add edges
        for dep_id in (dependencies or []):
            execution.graph.add_edge(dep_id, node.node_id)

        for dep_id in (dependents or []):
            execution.graph.add_edge(node.node_id, dep_id)

        # Assign agent
        if not node.assigned_agent and self._agents:
            node.assigned_agent = list(self._agents.keys())[0]

        logger.info(f"Node added to graph {graph_id}: {node.node_id}")

    async def remove_node_from_graph(
        self,
        graph_id: str,
        node_id: str
    ) -> None:
        """Remove a node from a graph (only if not running)."""
        execution = self._graphs.get(graph_id)
        if not execution:
            raise ValueError(f"Graph {graph_id} not found")

        node = execution.graph.get_node(node_id)
        if node.status == NodeStatus.RUNNING:
            raise RuntimeError("Cannot remove running node")

        execution.graph.remove_node(node_id)
        logger.info(f"Node removed from graph {graph_id}: {node_id}")

    # Query Methods

    def get_graph_status(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a graph execution."""
        execution = self._graphs.get(graph_id)
        if not execution:
            return None

        return {
            "graph_id": graph_id,
            "status": execution.status.value,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "nodes_total": len(execution.graph),
            "nodes_completed": sum(
                1 for n in execution.graph
                if n.status == NodeStatus.COMPLETED
            ),
            "nodes_failed": sum(
                1 for n in execution.graph
                if n.status == NodeStatus.FAILED
            ),
            "errors": execution.errors,
        }

    def get_all_graphs(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all graphs."""
        return {
            gid: self.get_graph_status(gid)
            for gid in self._graphs
        }

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent."""
        agent = self._agents.get(agent_id)
        if agent:
            return agent.get_info()
        return None

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all agents."""
        return {
            aid: agent.get_info()
            for aid, agent in self._agents.items()
        }

    @property
    def status(self) -> OrchestratorStatus:
        """Get orchestrator status."""
        return self._status
