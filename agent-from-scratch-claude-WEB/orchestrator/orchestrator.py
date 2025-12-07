"""
Task Graph Orchestrator with concurrent execution, worker autoscaling,
and dynamic graph mutation support.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple,
    Awaitable, TYPE_CHECKING
)
import logging

from core.types import (
    TaskNode, TaskGraph, TaskStatus, Message, MessageType,
    TaskExecutionError, GraphCycleError, generate_id, current_timestamp
)
from graph.task_graph import TaskGraphManager, TopologicalSorter, SelfHealingDAG
from agents.base_agent import BaseAgent, WorkerAgent, CoordinatorAgent

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics for task execution."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "running_tasks": self.running_tasks,
            "pending_tasks": self.pending_tasks,
            "success_rate": self.completed_tasks / max(1, self.total_tasks),
            "avg_execution_time": self.avg_execution_time,
            "total_execution_time": self.total_execution_time,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


@dataclass
class WorkerPool:
    """Pool of worker agents for task execution."""
    min_workers: int = 2
    max_workers: int = 10
    workers: Dict[str, WorkerAgent] = field(default_factory=dict)
    busy_workers: Set[str] = field(default_factory=set)
    idle_workers: Set[str] = field(default_factory=set)
    task_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    def __post_init__(self):
        self.task_queue = asyncio.Queue()


class WorkerAutoscaler:
    """Autoscaler for worker agents based on queue depth and execution metrics."""
    
    def __init__(
        self,
        pool: WorkerPool,
        worker_factory: Callable[[], WorkerAgent],
        scale_up_threshold: int = 5,
        scale_down_threshold: int = 1,
        scale_check_interval: float = 5.0,
        cooldown_period: float = 30.0
    ):
        self.pool = pool
        self.worker_factory = worker_factory
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        self.cooldown_period = cooldown_period
        
        self._last_scale_action = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the autoscaler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._autoscale_loop())
        
        while len(self.pool.workers) < self.pool.min_workers:
            await self._add_worker()
    
    async def stop(self):
        """Stop the autoscaler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _autoscale_loop(self):
        """Main autoscaling loop."""
        while self._running:
            await asyncio.sleep(self.scale_check_interval)
            await self._check_and_scale()
    
    async def _check_and_scale(self):
        """Check metrics and scale if needed."""
        now = time.time()
        
        if now - self._last_scale_action < self.cooldown_period:
            return
        
        queue_depth = self.pool.task_queue.qsize()
        idle_count = len(self.pool.idle_workers)
        total_workers = len(self.pool.workers)
        
        if queue_depth > self.scale_up_threshold and total_workers < self.pool.max_workers:
            workers_to_add = min(
                self.pool.max_workers - total_workers,
                (queue_depth - self.scale_up_threshold) // 2 + 1
            )
            for _ in range(workers_to_add):
                await self._add_worker()
            self._last_scale_action = now
            logger.info(f"Scaled up: added {workers_to_add} workers")
        
        elif idle_count > self.scale_down_threshold and total_workers > self.pool.min_workers:
            workers_to_remove = min(
                idle_count - self.scale_down_threshold,
                total_workers - self.pool.min_workers
            )
            for _ in range(workers_to_remove):
                await self._remove_idle_worker()
            self._last_scale_action = now
            logger.info(f"Scaled down: removed {workers_to_remove} workers")
    
    async def _add_worker(self):
        """Add a new worker to the pool."""
        worker = self.worker_factory()
        await worker.initialize()
        self.pool.workers[worker.id] = worker
        self.pool.idle_workers.add(worker.id)
    
    async def _remove_idle_worker(self):
        """Remove an idle worker from the pool."""
        if not self.pool.idle_workers:
            return
        worker_id = self.pool.idle_workers.pop()
        del self.pool.workers[worker_id]


class TaskOrchestrator:
    """Main task orchestrator with concurrent execution support."""
    
    def __init__(
        self,
        graph_manager: TaskGraphManager = None,
        max_concurrent_tasks: int = 10,
        worker_factory: Callable[[], WorkerAgent] = None
    ):
        self.graph_manager = graph_manager or TaskGraphManager()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.pool = WorkerPool(min_workers=2, max_workers=max_concurrent_tasks)
        
        if worker_factory is None:
            worker_factory = lambda: WorkerAgent(name=f"Worker-{generate_id()[:8]}")
        
        self.autoscaler = WorkerAutoscaler(self.pool, worker_factory)
        self.healing_dag = SelfHealingDAG(self.graph_manager)
        
        self._executing_graphs: Dict[str, asyncio.Task] = {}
        self._metrics: Dict[str, ExecutionMetrics] = {}
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the orchestrator."""
        await self.autoscaler.start()
        for _ in range(self.pool.min_workers):
            asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        """Stop the orchestrator."""
        await self.autoscaler.stop()
        for task in self._executing_graphs.values():
            task.cancel()
        self._thread_pool.shutdown(wait=False)
    
    async def execute_graph(
        self,
        graph_id: str,
        timeout: float = None
    ) -> ExecutionMetrics:
        """Execute a task graph with concurrent task execution."""
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph:
            raise ValueError(f"Graph {graph_id} not found")
        
        is_valid, error = await self.graph_manager.validate_graph(graph_id)
        if not is_valid:
            raise GraphCycleError(error)
        
        metrics = ExecutionMetrics(
            total_tasks=len(graph.nodes),
            pending_tasks=len(graph.nodes),
            start_time=current_timestamp()
        )
        self._metrics[graph_id] = metrics
        
        graph.status = TaskStatus.RUNNING
        graph.started_at = current_timestamp()
        
        try:
            levels = await self.graph_manager.get_execution_plan(graph_id)
            
            for level_idx, level_nodes in enumerate(levels):
                logger.info(f"Executing level {level_idx + 1}/{len(levels)} with {len(level_nodes)} tasks")
                
                tasks = []
                for node_id in level_nodes:
                    node = graph.nodes[node_id]
                    task = asyncio.create_task(
                        self._execute_node_with_retry(graph_id, node)
                    )
                    tasks.append(task)
                
                if timeout:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                else:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        metrics.failed_tasks += 1
                    elif isinstance(result, TaskNode) and result.status == TaskStatus.FAILED:
                        metrics.failed_tasks += 1
                    else:
                        metrics.completed_tasks += 1
                
                metrics.pending_tasks = metrics.total_tasks - metrics.completed_tasks - metrics.failed_tasks
            
            if metrics.failed_tasks > 0:
                graph.status = TaskStatus.FAILED
            else:
                graph.status = TaskStatus.COMPLETED
            
        except asyncio.TimeoutError:
            graph.status = TaskStatus.FAILED
            logger.error(f"Graph {graph_id} execution timed out")
        except Exception as e:
            graph.status = TaskStatus.FAILED
            logger.error(f"Graph {graph_id} execution failed: {e}")
            raise
        finally:
            graph.completed_at = current_timestamp()
            metrics.end_time = current_timestamp()
            
            if metrics.completed_tasks > 0:
                metrics.avg_execution_time = (
                    metrics.total_execution_time / metrics.completed_tasks
                )
            
            await self._trigger_callbacks(graph_id, "completed", metrics)
        
        return metrics
    
    async def _execute_node_with_retry(
        self,
        graph_id: str,
        node: TaskNode
    ) -> TaskNode:
        """Execute a node with retry support."""
        while node.retry_count <= node.max_retries:
            try:
                await self._prepare_node_inputs(graph_id, node)
                worker = await self._get_worker()
                
                try:
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        worker.execute_task(node),
                        timeout=node.timeout_seconds
                    )
                    execution_time = time.time() - start_time
                    
                    metrics = self._metrics.get(graph_id)
                    if metrics:
                        metrics.total_execution_time += execution_time
                    
                    await self.graph_manager.update_node_status(
                        graph_id,
                        node.id,
                        result.status,
                        result.output_data,
                        result.error_message
                    )
                    
                    if result.status == TaskStatus.COMPLETED:
                        return result
                    
                finally:
                    await self._return_worker(worker)
                
            except asyncio.TimeoutError:
                node.error_message = f"Task timed out after {node.timeout_seconds}s"
            except Exception as e:
                node.error_message = str(e)
            
            healed, action = await self.healing_dag.handle_failure(
                graph_id, node.id, Exception(node.error_message)
            )
            
            if not healed:
                break
            
            if action.startswith("retrying"):
                continue
            else:
                break
        
        node.status = TaskStatus.FAILED
        await self.graph_manager.update_node_status(
            graph_id, node.id, TaskStatus.FAILED, error_message=node.error_message
        )
        return node
    
    async def _prepare_node_inputs(self, graph_id: str, node: TaskNode):
        """Prepare node inputs from dependency outputs."""
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph:
            return
        
        for dep_id in node.dependencies:
            if dep_id in graph.nodes:
                dep_node = graph.nodes[dep_id]
                if dep_node.output_data:
                    node.input_data[f"dep_{dep_id}"] = dep_node.output_data
    
    async def _get_worker(self) -> WorkerAgent:
        """Get an available worker from the pool."""
        while True:
            async with self._lock:
                if self.pool.idle_workers:
                    worker_id = self.pool.idle_workers.pop()
                    self.pool.busy_workers.add(worker_id)
                    return self.pool.workers[worker_id]
            await asyncio.sleep(0.1)
    
    async def _return_worker(self, worker: WorkerAgent):
        """Return a worker to the pool."""
        async with self._lock:
            if worker.id in self.pool.busy_workers:
                self.pool.busy_workers.remove(worker.id)
            if worker.id in self.pool.workers:
                self.pool.idle_workers.add(worker.id)
    
    async def _worker_loop(self):
        """Worker loop for processing tasks from queue."""
        while True:
            try:
                task_info = await self.pool.task_queue.get()
                graph_id, node = task_info
                await self._execute_node_with_retry(graph_id, node)
                self.pool.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    async def add_node_to_running_graph(
        self,
        graph_id: str,
        name: str,
        description: str = "",
        dependencies: List[str] = None,
        input_data: Dict[str, Any] = None,
        priority: int = 5
    ) -> Optional[TaskNode]:
        """Add a node to a running graph (dynamic mutation)."""
        async with self._lock:
            graph = await self.graph_manager.get_graph(graph_id)
            if not graph or graph.status != TaskStatus.RUNNING:
                return None
            
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in graph.nodes:
                        raise ValueError(f"Dependency {dep_id} not found")
                    if graph.nodes[dep_id].status != TaskStatus.COMPLETED:
                        raise ValueError(f"Dependency {dep_id} not completed")
            
            node = await self.graph_manager.add_node(
                graph_id,
                name=name,
                description=description,
                dependencies=dependencies,
                input_data=input_data,
                priority=priority
            )
            
            if graph_id in self._metrics:
                self._metrics[graph_id].total_tasks += 1
                self._metrics[graph_id].pending_tasks += 1
            
            await self.pool.task_queue.put((graph_id, node))
            return node
    
    async def remove_pending_node(self, graph_id: str, node_id: str) -> bool:
        """Remove a pending node from a running graph."""
        async with self._lock:
            graph = await self.graph_manager.get_graph(graph_id)
            if not graph:
                return False
            
            node = graph.nodes.get(node_id)
            if not node or node.status != TaskStatus.PENDING:
                return False
            
            success = await self.graph_manager.remove_node(graph_id, node_id)
            
            if success and graph_id in self._metrics:
                self._metrics[graph_id].total_tasks -= 1
                self._metrics[graph_id].pending_tasks -= 1
            
            return success
    
    def on_event(self, event_type: str, callback: Callable):
        """Register a callback for an event type."""
        self._callbacks[event_type].append(callback)
    
    async def _trigger_callbacks(self, graph_id: str, event_type: str, data: Any):
        """Trigger callbacks for an event."""
        for callback in self._callbacks[event_type]:
            try:
                await callback(graph_id, event_type, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_metrics(self, graph_id: str = None) -> Dict[str, Any]:
        """Get execution metrics."""
        if graph_id:
            metrics = self._metrics.get(graph_id)
            return metrics.to_dict() if metrics else {}
        return {gid: m.to_dict() for gid, m in self._metrics.items()}
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get worker pool status."""
        return {
            "total_workers": len(self.pool.workers),
            "busy_workers": len(self.pool.busy_workers),
            "idle_workers": len(self.pool.idle_workers),
            "queue_depth": self.pool.task_queue.qsize(),
            "min_workers": self.pool.min_workers,
            "max_workers": self.pool.max_workers
        }


class DistributedOrchestrator(TaskOrchestrator):
    """Extended orchestrator with distributed execution support."""
    
    def __init__(self, message_queue=None, **kwargs):
        super().__init__(**kwargs)
        self.message_queue = message_queue
        self._remote_workers: Dict[str, str] = {}
    
    async def register_remote_worker(self, worker_id: str, queue_name: str):
        """Register a remote worker."""
        self._remote_workers[worker_id] = queue_name
    
    async def dispatch_to_remote(self, worker_id: str, task: TaskNode) -> bool:
        """Dispatch a task to a remote worker."""
        if not self.message_queue:
            return False
        
        queue_name = self._remote_workers.get(worker_id)
        if not queue_name:
            return False
        
        message = Message(
            type=MessageType.TASK_ASSIGNMENT,
            sender_id="orchestrator",
            receiver_id=worker_id,
            payload=task.to_dict()
        )
        
        return await self.message_queue.publish(queue_name, message)