"""
Task executor with concurrency support using asyncio and thread pools.
"""

from __future__ import annotations
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from ..core.graph import TaskNode, TaskGraph, NodeStatus
from ..core.message import Message, MessageType

if TYPE_CHECKING:
    from ..core.agent import Agent


class ExecutionMode(Enum):
    """Execution modes for tasks."""
    ASYNC = "async"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"


@dataclass
class ExecutionResult:
    """Result of task execution."""
    node_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    agent_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "retries": self.retries,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
        }


class TaskExecutor:
    """
    Executes tasks with support for concurrent execution using
    asyncio, thread pools, and process pools.
    """

    def __init__(
        self,
        max_workers: int = 10,
        max_process_workers: int = 4,
        default_mode: ExecutionMode = ExecutionMode.ASYNC,
        default_timeout: float = 300.0,
    ):
        self.max_workers = max_workers
        self.max_process_workers = max_process_workers
        self.default_mode = default_mode
        self.default_timeout = default_timeout

        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._agents: Dict[str, Agent] = {}
        self._results: Dict[str, ExecutionResult] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
            "on_error": [],
            "on_retry": [],
        }

    async def initialize(self) -> None:
        """Initialize executor resources."""
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=self.max_process_workers)
        self._semaphore = asyncio.Semaphore(self.max_workers)

    async def shutdown(self) -> None:
        """Shutdown executor and release resources."""
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()

        # Wait for cancellation
        if self._running_tasks:
            await asyncio.gather(
                *self._running_tasks.values(),
                return_exceptions=True
            )

        # Shutdown pools
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent for task execution."""
        self._agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_id, None)

    def add_hook(
        self,
        hook_type: str,
        callback: Callable
    ) -> None:
        """Add execution hook."""
        if hook_type in self._hooks:
            self._hooks[hook_type].append(callback)

    async def _run_hooks(
        self,
        hook_type: str,
        **kwargs: Any
    ) -> None:
        """Run all hooks of a specific type."""
        for hook in self._hooks.get(hook_type, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception:
                pass  # Hooks should not affect execution

    async def execute_node(
        self,
        node: TaskNode,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[ExecutionMode] = None
    ) -> ExecutionResult:
        """
        Execute a single task node.
        """
        mode = mode or self.default_mode
        timeout = node.timeout or self.default_timeout
        context = context or {}

        await self._run_hooks("pre_execute", node=node, context=context)

        node.mark_started()
        start_time = asyncio.get_event_loop().time()

        try:
            async with self._semaphore:
                if mode == ExecutionMode.ASYNC:
                    result = await self._execute_async(node, context, timeout)
                elif mode == ExecutionMode.THREAD_POOL:
                    result = await self._execute_in_thread(node, context, timeout)
                else:
                    result = await self._execute_in_process(node, context, timeout)

            execution_time = asyncio.get_event_loop().time() - start_time
            node.mark_completed(result)

            exec_result = ExecutionResult(
                node_id=node.node_id,
                success=True,
                result=result,
                execution_time=execution_time,
                retries=node.retry_count,
                agent_id=node.assigned_agent,
            )

            await self._run_hooks("post_execute", node=node, result=exec_result)
            self._results[node.node_id] = exec_result
            return exec_result

        except asyncio.TimeoutError:
            error = f"Task timed out after {timeout} seconds"
            return await self._handle_execution_error(node, error, start_time)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return await self._handle_execution_error(node, error, start_time)

    async def _handle_execution_error(
        self,
        node: TaskNode,
        error: str,
        start_time: float
    ) -> ExecutionResult:
        """Handle execution error with retry logic."""
        execution_time = asyncio.get_event_loop().time() - start_time

        if node.can_retry():
            node.retry_count += 1
            node.status = NodeStatus.RETRYING
            await self._run_hooks("on_retry", node=node, error=error)
            # Will be retried by orchestrator
        else:
            node.mark_failed(error)

        exec_result = ExecutionResult(
            node_id=node.node_id,
            success=False,
            error=error,
            execution_time=execution_time,
            retries=node.retry_count,
            agent_id=node.assigned_agent,
        )

        await self._run_hooks("on_error", node=node, error=error)
        self._results[node.node_id] = exec_result
        return exec_result

    async def _execute_async(
        self,
        node: TaskNode,
        context: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Execute task asynchronously."""
        agent = self._agents.get(node.assigned_agent)

        if agent:
            # Create task message
            message = Message(
                message_type=MessageType.TASK,
                sender_id="executor",
                receiver_id=agent.agent_id,
                payload={
                    "task": node.task_data,
                    "type": node.task_type,
                    "context": context,
                    "node_id": node.node_id,
                }
            )

            await agent.receive_message(message)

            # Wait for result with timeout
            result = await asyncio.wait_for(
                self._wait_for_agent_result(agent, message.message_id),
                timeout=timeout
            )
            return result
        else:
            # Execute locally if no agent assigned
            return await self._execute_local(node, context)

    async def _wait_for_agent_result(
        self,
        agent: Agent,
        message_id: str
    ) -> Dict[str, Any]:
        """Wait for agent to produce a result."""
        while True:
            # Check outbox for response
            for i, msg in enumerate(agent.outbox._queue):
                if msg.correlation_id == message_id:
                    agent.outbox._queue.pop(i)
                    if msg.message_type == MessageType.ERROR:
                        raise Exception(msg.payload.get("error", "Unknown error"))
                    return msg.payload.get("result", {})
            await asyncio.sleep(0.01)

    async def _execute_local(
        self,
        node: TaskNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task locally without an agent."""
        return {
            "status": "completed",
            "task_type": node.task_type,
            "task_data": node.task_data,
            "context_used": bool(context),
        }

    async def _execute_in_thread(
        self,
        node: TaskNode,
        context: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Execute task in thread pool."""
        loop = asyncio.get_event_loop()

        def sync_task():
            return {
                "status": "completed",
                "task_type": node.task_type,
                "task_data": node.task_data,
                "executed_in": "thread_pool",
            }

        result = await asyncio.wait_for(
            loop.run_in_executor(self._thread_pool, sync_task),
            timeout=timeout
        )
        return result

    async def _execute_in_process(
        self,
        node: TaskNode,
        context: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Execute task in process pool."""
        loop = asyncio.get_event_loop()

        # Note: Functions must be picklable for process pool
        result = await asyncio.wait_for(
            loop.run_in_executor(
                self._process_pool,
                _process_task,
                node.to_dict(),
                context
            ),
            timeout=timeout
        )
        return result

    async def execute_parallel(
        self,
        nodes: List[TaskNode],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionResult]:
        """Execute multiple nodes in parallel."""
        tasks = [
            self.execute_node(node, context)
            for node in nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    node_id=nodes[i].node_id,
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def execute_graph(
        self,
        graph: TaskGraph,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ExecutionResult]:
        """
        Execute all tasks in a graph respecting dependencies.
        Uses parallel execution for independent nodes.
        """
        context = context or {}
        results: Dict[str, ExecutionResult] = {}

        # Get parallel execution groups
        parallel_groups = graph.get_parallel_groups()

        for group in parallel_groups:
            # Get nodes that are ready to execute
            ready_nodes = [
                graph.get_node(node_id)
                for node_id in group
                if graph.get_node(node_id).status in (NodeStatus.PENDING, NodeStatus.READY)
            ]

            if not ready_nodes:
                continue

            # Build context from completed dependencies
            group_context = context.copy()
            for node in ready_nodes:
                for parent in graph.get_parents(node.node_id):
                    if parent.result:
                        group_context[f"result_{parent.node_id}"] = parent.result

            # Execute in parallel
            group_results = await self.execute_parallel(ready_nodes, group_context)

            for result in group_results:
                results[result.node_id] = result

                # Handle failures
                if not result.success:
                    node = graph.get_node(result.node_id)
                    if node.status == NodeStatus.RETRYING:
                        # Re-add to execution
                        pass

        return results

    def get_result(self, node_id: str) -> Optional[ExecutionResult]:
        """Get result for a specific node."""
        return self._results.get(node_id)

    def get_all_results(self) -> Dict[str, ExecutionResult]:
        """Get all execution results."""
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()


def _process_task(
    node_dict: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process task in separate process.
    This function must be at module level to be picklable.
    """
    return {
        "status": "completed",
        "task_type": node_dict["task_type"],
        "task_data": node_dict["task_data"],
        "executed_in": "process_pool",
    }
