"""
Task Graph module with DAG operations, cycle detection, topological ordering,
and dynamic graph mutation support.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, 
    Awaitable, Iterator, TYPE_CHECKING
)
import logging

from core.types import (
    TaskNode, TaskGraph, TaskStatus, GraphCycleError,
    TaskExecutionError, generate_id, current_timestamp
)

logger = logging.getLogger(__name__)


class CycleDetector:
    """
    Detects cycles in directed graphs using DFS-based algorithm.
    """
    
    @staticmethod
    def detect_cycle(
        nodes: Dict[str, TaskNode],
        edges: List[Tuple[str, str]]
    ) -> Optional[List[str]]:
        """
        Detect if there's a cycle in the graph.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            edges: List of (from_id, to_id) tuples
            
        Returns:
            List of node IDs forming a cycle if found, None otherwise
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = defaultdict(list)
        for from_id, to_id in edges:
            adj[from_id].append(to_id)
        
        # Colors: 0=white (unvisited), 1=gray (in progress), 2=black (done)
        colors: Dict[str, int] = {node_id: 0 for node_id in nodes}
        parent: Dict[str, Optional[str]] = {node_id: None for node_id in nodes}
        cycle_start: Optional[str] = None
        cycle_end: Optional[str] = None
        
        def dfs(node: str) -> bool:
            nonlocal cycle_start, cycle_end
            colors[node] = 1
            
            for neighbor in adj[node]:
                if neighbor not in colors:
                    continue
                    
                if colors[neighbor] == 0:
                    parent[neighbor] = node
                    if dfs(neighbor):
                        return True
                elif colors[neighbor] == 1:
                    # Found cycle
                    cycle_start = neighbor
                    cycle_end = node
                    return True
            
            colors[node] = 2
            return False
        
        # Run DFS from each unvisited node
        for node_id in nodes:
            if colors[node_id] == 0:
                if dfs(node_id):
                    # Reconstruct cycle path
                    cycle: List[str] = [cycle_start]
                    current = cycle_end
                    while current != cycle_start:
                        cycle.append(current)
                        current = parent.get(current)
                    cycle.append(cycle_start)
                    cycle.reverse()
                    return cycle
        
        return None


class TopologicalSorter:
    """
    Performs topological sort on directed acyclic graphs.
    Uses Kahn's algorithm for iterative sorting.
    """
    
    @staticmethod
    def sort(
        nodes: Dict[str, TaskNode],
        edges: List[Tuple[str, str]]
    ) -> List[str]:
        """
        Perform topological sort on the graph.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            edges: List of (from_id, to_id) tuples
            
        Returns:
            List of node IDs in topological order
            
        Raises:
            GraphCycleError: If the graph contains a cycle
        """
        # Build adjacency list and in-degree count
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {node_id: 0 for node_id in nodes}
        
        for from_id, to_id in edges:
            if from_id in nodes and to_id in nodes:
                adj[from_id].append(to_id)
                in_degree[to_id] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([
            node_id for node_id, degree in in_degree.items() 
            if degree == 0
        ])
        
        result: List[str] = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(nodes):
            # Cycle detected
            remaining = set(nodes.keys()) - set(result)
            raise GraphCycleError(
                f"Graph contains a cycle involving nodes: {remaining}"
            )
        
        return result
    
    @staticmethod
    def get_execution_levels(
        nodes: Dict[str, TaskNode],
        edges: List[Tuple[str, str]]
    ) -> List[List[str]]:
        """
        Group nodes into execution levels where nodes in the same level
        can be executed in parallel.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            edges: List of (from_id, to_id) tuples
            
        Returns:
            List of lists, where each inner list contains node IDs
            that can be executed in parallel
        """
        # Build reverse adjacency (dependencies)
        dependencies: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
        for from_id, to_id in edges:
            if from_id in nodes and to_id in nodes:
                dependencies[to_id].add(from_id)
        
        completed: Set[str] = set()
        levels: List[List[str]] = []
        remaining = set(nodes.keys())
        
        while remaining:
            # Find all nodes whose dependencies are satisfied
            ready = [
                node_id for node_id in remaining
                if dependencies[node_id].issubset(completed)
            ]
            
            if not ready:
                raise GraphCycleError("Cycle detected: no ready nodes found")
            
            levels.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return levels


class TaskGraphManager:
    """
    Manages task graphs with support for dynamic mutation, execution tracking,
    and parallel execution planning.
    """
    
    def __init__(self):
        self._graphs: Dict[str, TaskGraph] = {}
        self._lock = asyncio.Lock()
    
    async def create_graph(
        self,
        name: str,
        description: str = "",
        owner_id: str = ""
    ) -> TaskGraph:
        """Create a new empty task graph."""
        async with self._lock:
            graph = TaskGraph(
                id=generate_id(),
                name=name,
                description=description,
                owner_id=owner_id,
                status=TaskStatus.PENDING
            )
            self._graphs[graph.id] = graph
            return graph
    
    async def get_graph(self, graph_id: str) -> Optional[TaskGraph]:
        """Get a graph by ID."""
        return self._graphs.get(graph_id)
    
    async def add_node(
        self,
        graph_id: str,
        name: str,
        description: str = "",
        dependencies: List[str] = None,
        input_data: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        priority: int = 5,
        timeout_seconds: int = 300
    ) -> TaskNode:
        """
        Add a new node to a graph.
        
        Args:
            graph_id: ID of the graph
            name: Name of the task
            description: Task description
            dependencies: List of node IDs this task depends on
            input_data: Input data for the task
            metadata: Additional metadata
            priority: Task priority (1-10)
            timeout_seconds: Task timeout
            
        Returns:
            Created TaskNode
            
        Raises:
            GraphCycleError: If adding the node would create a cycle
        """
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} not found")
            
            node = TaskNode(
                id=generate_id(),
                name=name,
                description=description,
                dependencies=dependencies or [],
                input_data=input_data or {},
                metadata=metadata or {},
                priority=priority,
                timeout_seconds=timeout_seconds
            )
            
            # Validate dependencies exist
            for dep_id in node.dependencies:
                if dep_id not in graph.nodes:
                    raise ValueError(f"Dependency {dep_id} not found in graph")
            
            # Add node and edges temporarily
            graph.nodes[node.id] = node
            new_edges = graph.edges.copy()
            for dep_id in node.dependencies:
                new_edges.append((dep_id, node.id))
            
            # Check for cycles
            cycle = CycleDetector.detect_cycle(graph.nodes, new_edges)
            if cycle:
                del graph.nodes[node.id]
                raise GraphCycleError(f"Adding node would create cycle: {cycle}")
            
            # Commit edges
            graph.edges = new_edges
            
            return node
    
    async def add_edge(
        self,
        graph_id: str,
        from_node_id: str,
        to_node_id: str
    ) -> bool:
        """
        Add an edge between two nodes.
        
        Args:
            graph_id: ID of the graph
            from_node_id: Source node ID
            to_node_id: Target node ID
            
        Returns:
            True if edge was added
            
        Raises:
            GraphCycleError: If adding the edge would create a cycle
        """
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph:
                raise ValueError(f"Graph {graph_id} not found")
            
            if from_node_id not in graph.nodes:
                raise ValueError(f"Source node {from_node_id} not found")
            if to_node_id not in graph.nodes:
                raise ValueError(f"Target node {to_node_id} not found")
            
            # Check if edge already exists
            if (from_node_id, to_node_id) in graph.edges:
                return False
            
            # Check for cycle
            new_edges = graph.edges + [(from_node_id, to_node_id)]
            cycle = CycleDetector.detect_cycle(graph.nodes, new_edges)
            if cycle:
                raise GraphCycleError(f"Adding edge would create cycle: {cycle}")
            
            graph.edges = new_edges
            graph.nodes[to_node_id].dependencies.append(from_node_id)
            
            return True
    
    async def remove_node(
        self,
        graph_id: str,
        node_id: str,
        cascade: bool = False
    ) -> bool:
        """
        Remove a node from the graph.
        
        Args:
            graph_id: ID of the graph
            node_id: ID of the node to remove
            cascade: If True, also remove dependent nodes
            
        Returns:
            True if node was removed
        """
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph:
                return False
            
            if node_id not in graph.nodes:
                return False
            
            # Find dependent nodes
            dependents = [
                nid for nid, node in graph.nodes.items()
                if node_id in node.dependencies
            ]
            
            if dependents and not cascade:
                raise ValueError(
                    f"Node {node_id} has dependents: {dependents}. "
                    "Use cascade=True to remove them."
                )
            
            # Remove node and its edges
            del graph.nodes[node_id]
            graph.edges = [
                (f, t) for f, t in graph.edges
                if f != node_id and t != node_id
            ]
            
            # Update dependencies in other nodes
            for node in graph.nodes.values():
                if node_id in node.dependencies:
                    node.dependencies.remove(node_id)
            
            # Cascade removal
            if cascade:
                for dep_id in dependents:
                    await self.remove_node(graph_id, dep_id, cascade=True)
            
            return True
    
    async def remove_edge(
        self,
        graph_id: str,
        from_node_id: str,
        to_node_id: str
    ) -> bool:
        """Remove an edge from the graph."""
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph:
                return False
            
            edge = (from_node_id, to_node_id)
            if edge not in graph.edges:
                return False
            
            graph.edges.remove(edge)
            if from_node_id in graph.nodes[to_node_id].dependencies:
                graph.nodes[to_node_id].dependencies.remove(from_node_id)
            
            return True
    
    async def update_node(
        self,
        graph_id: str,
        node_id: str,
        updates: Dict[str, Any]
    ) -> Optional[TaskNode]:
        """Update node properties."""
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph or node_id not in graph.nodes:
                return None
            
            node = graph.nodes[node_id]
            allowed_fields = {
                'name', 'description', 'input_data', 'metadata',
                'priority', 'timeout_seconds', 'max_retries'
            }
            
            for key, value in updates.items():
                if key in allowed_fields:
                    setattr(node, key, value)
            
            return node
    
    async def update_node_status(
        self,
        graph_id: str,
        node_id: str,
        status: TaskStatus,
        output_data: Dict[str, Any] = None,
        error_message: str = None
    ) -> Optional[TaskNode]:
        """Update node execution status."""
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph or node_id not in graph.nodes:
                return None
            
            node = graph.nodes[node_id]
            node.status = status
            
            if status == TaskStatus.RUNNING:
                node.started_at = current_timestamp()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                node.completed_at = current_timestamp()
            
            if output_data is not None:
                node.output_data = output_data
            if error_message is not None:
                node.error_message = error_message
            
            return node
    
    async def get_ready_nodes(self, graph_id: str) -> List[TaskNode]:
        """
        Get all nodes that are ready to execute
        (all dependencies completed).
        """
        graph = self._graphs.get(graph_id)
        if not graph:
            return []
        
        ready = []
        for node in graph.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            
            # Check all dependencies are completed
            deps_satisfied = all(
                graph.nodes[dep_id].status == TaskStatus.COMPLETED
                for dep_id in node.dependencies
                if dep_id in graph.nodes
            )
            
            if deps_satisfied:
                ready.append(node)
        
        # Sort by priority (higher first)
        ready.sort(key=lambda n: n.priority, reverse=True)
        return ready
    
    async def get_execution_plan(self, graph_id: str) -> List[List[str]]:
        """
        Get the execution plan as parallel execution levels.
        
        Returns:
            List of levels, where each level contains node IDs
            that can be executed in parallel
        """
        graph = self._graphs.get(graph_id)
        if not graph:
            return []
        
        return TopologicalSorter.get_execution_levels(
            graph.nodes, graph.edges
        )
    
    async def get_topological_order(self, graph_id: str) -> List[str]:
        """Get nodes in topological order."""
        graph = self._graphs.get(graph_id)
        if not graph:
            return []
        
        return TopologicalSorter.sort(graph.nodes, graph.edges)
    
    async def validate_graph(self, graph_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a graph for cycles and completeness.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        graph = self._graphs.get(graph_id)
        if not graph:
            return False, "Graph not found"
        
        # Check for cycles
        cycle = CycleDetector.detect_cycle(graph.nodes, graph.edges)
        if cycle:
            return False, f"Graph contains cycle: {cycle}"
        
        # Check for dangling dependencies
        for node in graph.nodes.values():
            for dep_id in node.dependencies:
                if dep_id not in graph.nodes:
                    return False, f"Node {node.id} has invalid dependency {dep_id}"
        
        return True, None
    
    async def clone_graph(
        self,
        graph_id: str,
        new_name: str = None
    ) -> Optional[TaskGraph]:
        """Clone a graph with new IDs."""
        async with self._lock:
            graph = self._graphs.get(graph_id)
            if not graph:
                return None
            
            # Create ID mapping
            id_map: Dict[str, str] = {}
            for old_id in graph.nodes:
                id_map[old_id] = generate_id()
            
            # Clone nodes with new IDs
            new_nodes: Dict[str, TaskNode] = {}
            for old_id, node in graph.nodes.items():
                new_node = TaskNode(
                    id=id_map[old_id],
                    name=node.name,
                    description=node.description,
                    dependencies=[id_map[d] for d in node.dependencies if d in id_map],
                    input_data=node.input_data.copy(),
                    metadata=node.metadata.copy(),
                    priority=node.priority,
                    timeout_seconds=node.timeout_seconds,
                    max_retries=node.max_retries
                )
                new_nodes[new_node.id] = new_node
            
            # Clone edges with new IDs
            new_edges = [
                (id_map[f], id_map[t])
                for f, t in graph.edges
                if f in id_map and t in id_map
            ]
            
            new_graph = TaskGraph(
                id=generate_id(),
                name=new_name or f"{graph.name} (copy)",
                description=graph.description,
                nodes=new_nodes,
                edges=new_edges,
                owner_id=graph.owner_id,
                metadata=graph.metadata.copy()
            )
            
            self._graphs[new_graph.id] = new_graph
            return new_graph
    
    async def merge_graphs(
        self,
        graph_ids: List[str],
        new_name: str,
        connections: List[Tuple[str, str, str, str]] = None
    ) -> Optional[TaskGraph]:
        """
        Merge multiple graphs into one.
        
        Args:
            graph_ids: List of graph IDs to merge
            new_name: Name for the new graph
            connections: Optional list of (graph1_id, node1_id, graph2_id, node2_id)
                        to connect graphs
            
        Returns:
            Merged TaskGraph
        """
        async with self._lock:
            # Collect all graphs
            graphs = [self._graphs.get(gid) for gid in graph_ids]
            if not all(graphs):
                return None
            
            # Create ID mapping for each graph
            id_maps: Dict[str, Dict[str, str]] = {}
            for graph in graphs:
                id_maps[graph.id] = {}
                for old_id in graph.nodes:
                    id_maps[graph.id][old_id] = generate_id()
            
            # Merge nodes
            new_nodes: Dict[str, TaskNode] = {}
            for graph in graphs:
                id_map = id_maps[graph.id]
                for old_id, node in graph.nodes.items():
                    new_node = TaskNode(
                        id=id_map[old_id],
                        name=f"{graph.name}/{node.name}",
                        description=node.description,
                        dependencies=[id_map[d] for d in node.dependencies if d in id_map],
                        input_data=node.input_data.copy(),
                        metadata={**node.metadata, "source_graph": graph.id},
                        priority=node.priority,
                        timeout_seconds=node.timeout_seconds
                    )
                    new_nodes[new_node.id] = new_node
            
            # Merge edges
            new_edges: List[Tuple[str, str]] = []
            for graph in graphs:
                id_map = id_maps[graph.id]
                for f, t in graph.edges:
                    if f in id_map and t in id_map:
                        new_edges.append((id_map[f], id_map[t]))
            
            # Add connections between graphs
            if connections:
                for g1_id, n1_id, g2_id, n2_id in connections:
                    if g1_id in id_maps and g2_id in id_maps:
                        new_from = id_maps[g1_id].get(n1_id)
                        new_to = id_maps[g2_id].get(n2_id)
                        if new_from and new_to:
                            new_edges.append((new_from, new_to))
                            new_nodes[new_to].dependencies.append(new_from)
            
            new_graph = TaskGraph(
                id=generate_id(),
                name=new_name,
                nodes=new_nodes,
                edges=new_edges,
                metadata={"merged_from": graph_ids}
            )
            
            # Validate no cycles
            cycle = CycleDetector.detect_cycle(new_nodes, new_edges)
            if cycle:
                raise GraphCycleError(f"Merged graph contains cycle: {cycle}")
            
            self._graphs[new_graph.id] = new_graph
            return new_graph
    
    async def get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """Get statistics about a graph."""
        graph = self._graphs.get(graph_id)
        if not graph:
            return {}
        
        status_counts = defaultdict(int)
        for node in graph.nodes.values():
            status_counts[node.status.value] += 1
        
        return {
            "id": graph.id,
            "name": graph.name,
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "status_counts": dict(status_counts),
            "is_valid": (await self.validate_graph(graph_id))[0],
            "execution_levels": len(await self.get_execution_plan(graph_id))
        }
    
    async def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph."""
        async with self._lock:
            if graph_id in self._graphs:
                del self._graphs[graph_id]
                return True
            return False


# ============================================================================
# SELF-HEALING DAG
# ============================================================================

class SelfHealingDAG:
    """
    Self-healing DAG that can recover from failures by
    re-routing, retrying, or skipping failed nodes.
    """
    
    def __init__(self, graph_manager: TaskGraphManager):
        self.graph_manager = graph_manager
        self._failure_handlers: Dict[str, Callable] = {}
        self._healing_strategies: Dict[str, str] = {}  # node_id -> strategy
    
    async def register_healing_strategy(
        self,
        graph_id: str,
        node_id: str,
        strategy: str,  # 'retry', 'skip', 'fallback', 'reroute'
        fallback_node_id: str = None
    ):
        """Register a healing strategy for a node."""
        key = f"{graph_id}:{node_id}"
        self._healing_strategies[key] = strategy
        if fallback_node_id:
            self._healing_strategies[f"{key}:fallback"] = fallback_node_id
    
    async def handle_failure(
        self,
        graph_id: str,
        node_id: str,
        error: Exception
    ) -> Tuple[bool, str]:
        """
        Handle a node failure according to healing strategy.
        
        Returns:
            Tuple of (healed, action_taken)
        """
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph or node_id not in graph.nodes:
            return False, "node_not_found"
        
        node = graph.nodes[node_id]
        key = f"{graph_id}:{node_id}"
        strategy = self._healing_strategies.get(key, 'retry')
        
        if strategy == 'retry':
            if node.retry_count < node.max_retries:
                node.retry_count += 1
                node.status = TaskStatus.RETRYING
                node.error_message = str(error)
                return True, f"retrying ({node.retry_count}/{node.max_retries})"
            else:
                node.status = TaskStatus.FAILED
                return False, "max_retries_exceeded"
        
        elif strategy == 'skip':
            node.status = TaskStatus.COMPLETED
            node.output_data = {"skipped": True, "reason": str(error)}
            return True, "skipped"
        
        elif strategy == 'fallback':
            fallback_id = self._healing_strategies.get(f"{key}:fallback")
            if fallback_id and fallback_id in graph.nodes:
                # Mark current as failed, activate fallback
                node.status = TaskStatus.FAILED
                fallback = graph.nodes[fallback_id]
                fallback.status = TaskStatus.READY
                return True, f"fallback_to_{fallback_id}"
            return False, "no_fallback_available"
        
        elif strategy == 'reroute':
            # Remove this node and reconnect dependencies
            dependents = [
                n for n in graph.nodes.values()
                if node_id in n.dependencies
            ]
            for dep in dependents:
                dep.dependencies.remove(node_id)
                dep.dependencies.extend(node.dependencies)
            
            node.status = TaskStatus.CANCELLED
            return True, "rerouted"
        
        return False, "unknown_strategy"
    
    async def get_healthy_path(
        self,
        graph_id: str,
        start_node_id: str,
        end_node_id: str
    ) -> Optional[List[str]]:
        """
        Find a healthy path between two nodes,
        avoiding failed nodes if possible.
        """
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph:
            return None
        
        # Build adjacency list excluding failed nodes
        adj: Dict[str, List[str]] = defaultdict(list)
        for from_id, to_id in graph.edges:
            if graph.nodes[from_id].status != TaskStatus.FAILED:
                adj[from_id].append(to_id)
        
        # BFS to find path
        queue = deque([(start_node_id, [start_node_id])])
        visited = {start_node_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_node_id:
                return path
            
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None