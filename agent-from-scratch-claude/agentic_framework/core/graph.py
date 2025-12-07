"""
Graph-based task orchestration with DAG support, cycle detection, and topological ordering.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .message import Message, MessageType


class NodeStatus(Enum):
    """Status of a task node."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class EdgeType(Enum):
    """Types of edges between nodes."""
    DEPENDENCY = "dependency"  # Target depends on source completing
    DATA_FLOW = "data_flow"    # Data passes from source to target
    CONDITIONAL = "conditional"  # Conditional execution
    PARALLEL = "parallel"      # Can run in parallel


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the graph."""
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Cycle detected: {' -> '.join(cycle)}")


class NodeNotFoundError(Exception):
    """Raised when a node is not found."""
    pass


@dataclass
class TaskNode:
    """
    A node in the task graph representing a single task.
    """
    node_id: str
    task_type: str
    task_data: Dict[str, Any]
    assigned_agent: Optional[str] = None
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    priority: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "node_id": self.node_id,
            "task_type": self.task_type,
            "task_data": self.task_data,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskNode:
        """Create node from dictionary."""
        data = data.copy()
        data["status"] = NodeStatus(data["status"])
        # Remove condition as it's not serializable
        data.pop("condition", None)
        return cls(**data)

    def can_retry(self) -> bool:
        """Check if node can be retried."""
        return self.retry_count < self.max_retries

    def mark_started(self) -> None:
        """Mark node as started."""
        self.status = NodeStatus.RUNNING
        self.started_at = datetime.utcnow().isoformat()

    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark node as completed."""
        self.status = NodeStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.utcnow().isoformat()

    def mark_failed(self, error: str) -> None:
        """Mark node as failed."""
        self.status = NodeStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow().isoformat()

    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at)
        return (end - start).total_seconds()


@dataclass
class Edge:
    """An edge connecting two nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DEPENDENCY
    condition: Optional[str] = None  # JSON-serializable condition
    data_mapping: Optional[Dict[str, str]] = None  # Maps source output to target input

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "condition": self.condition,
            "data_mapping": self.data_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        """Create edge from dictionary."""
        data = data.copy()
        data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


class TaskGraph:
    """
    Directed Acyclic Graph (DAG) for task orchestration.
    Supports cycle detection, topological ordering, and dynamic mutation.
    """

    def __init__(self, graph_id: Optional[str] = None):
        self.graph_id = graph_id or str(uuid.uuid4())
        self._nodes: Dict[str, TaskNode] = {}
        self._edges: List[Edge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # node -> children
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # node -> parents
        self._topological_order: Optional[List[str]] = None
        self._dirty = True  # Track if graph has changed
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.utcnow().isoformat()

    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self._nodes[node.node_id] = node
        self._dirty = True

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges from the graph."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        # Remove edges
        self._edges = [e for e in self._edges
                       if e.source_id != node_id and e.target_id != node_id]

        # Update adjacency
        for child in self._adjacency.get(node_id, set()).copy():
            self._reverse_adjacency[child].discard(node_id)
        for parent in self._reverse_adjacency.get(node_id, set()).copy():
            self._adjacency[parent].discard(node_id)

        del self._adjacency[node_id]
        del self._reverse_adjacency[node_id]
        del self._nodes[node_id]
        self._dirty = True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEPENDENCY,
        condition: Optional[str] = None,
        data_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """Add an edge between two nodes."""
        if source_id not in self._nodes:
            raise NodeNotFoundError(f"Source node {source_id} not found")
        if target_id not in self._nodes:
            raise NodeNotFoundError(f"Target node {target_id} not found")

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
            data_mapping=data_mapping
        )
        self._edges.append(edge)
        self._adjacency[source_id].add(target_id)
        self._reverse_adjacency[target_id].add(source_id)
        self._dirty = True

        # Check for cycles after adding edge
        if self._has_cycle():
            # Rollback
            self._edges.pop()
            self._adjacency[source_id].discard(target_id)
            self._reverse_adjacency[target_id].discard(source_id)
            cycle = self._find_cycle()
            raise CycleDetectedError(cycle or [source_id, target_id])

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove an edge between two nodes."""
        self._edges = [e for e in self._edges
                       if not (e.source_id == source_id and e.target_id == target_id)]
        self._adjacency[source_id].discard(target_id)
        self._reverse_adjacency[target_id].discard(source_id)
        self._dirty = True

    def get_node(self, node_id: str) -> TaskNode:
        """Get a node by ID."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        return self._nodes[node_id]

    def get_nodes(self) -> List[TaskNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_edges(self) -> List[Edge]:
        """Get all edges."""
        return self._edges.copy()

    def get_children(self, node_id: str) -> List[TaskNode]:
        """Get child nodes (dependencies)."""
        return [self._nodes[child_id] for child_id in self._adjacency.get(node_id, set())]

    def get_parents(self, node_id: str) -> List[TaskNode]:
        """Get parent nodes (dependents)."""
        return [self._nodes[parent_id] for parent_id in self._reverse_adjacency.get(node_id, set())]

    def _has_cycle(self) -> bool:
        """Check if graph has a cycle using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self._nodes}

        def dfs(node_id: str) -> bool:
            colors[node_id] = GRAY
            for child_id in self._adjacency.get(node_id, set()):
                if colors[child_id] == GRAY:  # Back edge found
                    return True
                if colors[child_id] == WHITE and dfs(child_id):
                    return True
            colors[node_id] = BLACK
            return False

        for node_id in self._nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False

    def _find_cycle(self) -> Optional[List[str]]:
        """Find and return a cycle if one exists."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self._nodes}
        parent: Dict[str, Optional[str]] = {node_id: None for node_id in self._nodes}
        cycle_start: Optional[str] = None

        def dfs(node_id: str) -> bool:
            nonlocal cycle_start
            colors[node_id] = GRAY
            for child_id in self._adjacency.get(node_id, set()):
                if colors[child_id] == GRAY:
                    parent[child_id] = node_id
                    cycle_start = child_id
                    return True
                if colors[child_id] == WHITE:
                    parent[child_id] = node_id
                    if dfs(child_id):
                        return True
            colors[node_id] = BLACK
            return False

        for node_id in self._nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    # Reconstruct cycle
                    cycle = [cycle_start]
                    current = parent.get(cycle_start)
                    while current and current != cycle_start:
                        cycle.append(current)
                        current = parent.get(current)
                    cycle.append(cycle_start)
                    return list(reversed(cycle))
        return None

    def topological_sort(self) -> List[str]:
        """
        Return nodes in topological order (Kahn's algorithm).
        Raises CycleDetectedError if graph has cycles.
        """
        if not self._dirty and self._topological_order is not None:
            return self._topological_order

        if self._has_cycle():
            cycle = self._find_cycle()
            raise CycleDetectedError(cycle or [])

        in_degree = {node_id: 0 for node_id in self._nodes}
        for node_id in self._nodes:
            for child_id in self._adjacency.get(node_id, set()):
                in_degree[child_id] += 1

        # Start with nodes that have no dependencies
        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            for child_id in self._adjacency.get(node_id, set()):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(result) != len(self._nodes):
            raise CycleDetectedError(self._find_cycle() or [])

        self._topological_order = result
        self._dirty = False
        return result

    def get_ready_nodes(self) -> List[TaskNode]:
        """Get nodes that are ready to execute (all dependencies completed)."""
        ready = []
        for node in self._nodes.values():
            if node.status == NodeStatus.PENDING:
                parents = self.get_parents(node.node_id)
                if all(p.status == NodeStatus.COMPLETED for p in parents):
                    node.status = NodeStatus.READY
                    ready.append(node)
            elif node.status == NodeStatus.READY:
                ready.append(node)
        return sorted(ready, key=lambda n: -n.priority)

    def get_parallel_groups(self) -> List[List[str]]:
        """
        Get groups of nodes that can be executed in parallel.
        Returns list of lists, where each inner list contains
        node IDs that can run concurrently.
        """
        order = self.topological_sort()
        levels: Dict[str, int] = {}

        for node_id in order:
            parents = self._reverse_adjacency.get(node_id, set())
            if not parents:
                levels[node_id] = 0
            else:
                levels[node_id] = max(levels[p] + 1 for p in parents)

        # Group by level
        groups: Dict[int, List[str]] = defaultdict(list)
        for node_id, level in levels.items():
            groups[level].append(node_id)

        return [groups[i] for i in sorted(groups.keys())]

    def get_critical_path(self) -> List[str]:
        """
        Find the critical path (longest path) through the graph.
        Useful for estimating minimum completion time.
        """
        order = self.topological_sort()
        distances: Dict[str, float] = {node_id: 0 for node_id in self._nodes}
        predecessors: Dict[str, Optional[str]] = {node_id: None for node_id in self._nodes}

        for node_id in order:
            node = self._nodes[node_id]
            node_time = node.timeout  # Use timeout as estimated duration

            for child_id in self._adjacency.get(node_id, set()):
                new_dist = distances[node_id] + node_time
                if new_dist > distances[child_id]:
                    distances[child_id] = new_dist
                    predecessors[child_id] = node_id

        # Find the end node with maximum distance
        end_node = max(distances.keys(), key=lambda k: distances[k])

        # Reconstruct path
        path = []
        current: Optional[str] = end_node
        while current is not None:
            path.append(current)
            current = predecessors[current]

        return list(reversed(path))

    def validate(self) -> List[str]:
        """Validate graph and return list of issues."""
        issues = []

        # Check for cycles
        if self._has_cycle():
            issues.append("Graph contains cycles")

        # Check for orphan nodes (no edges at all)
        for node_id in self._nodes:
            if (not self._adjacency.get(node_id) and
                not self._reverse_adjacency.get(node_id) and
                len(self._nodes) > 1):
                issues.append(f"Node {node_id} is orphaned (no edges)")

        # Check for missing agent assignments
        for node in self._nodes.values():
            if not node.assigned_agent:
                issues.append(f"Node {node.node_id} has no assigned agent")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskGraph:
        """Deserialize graph from dictionary."""
        graph = cls(graph_id=data["graph_id"])
        graph.metadata = data.get("metadata", {})
        graph.created_at = data.get("created_at", datetime.utcnow().isoformat())

        for node_data in data["nodes"]:
            graph.add_node(TaskNode.from_dict(node_data))

        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data)
            graph._edges.append(edge)
            graph._adjacency[edge.source_id].add(edge.target_id)
            graph._reverse_adjacency[edge.target_id].add(edge.source_id)

        return graph

    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> TaskGraph:
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def clone(self) -> TaskGraph:
        """Create a deep copy of the graph."""
        return TaskGraph.from_dict(self.to_dict())

    def subgraph(self, node_ids: Set[str]) -> TaskGraph:
        """Create a subgraph containing only specified nodes."""
        subgraph = TaskGraph()

        for node_id in node_ids:
            if node_id in self._nodes:
                # Create a copy of the node
                node_data = self._nodes[node_id].to_dict()
                subgraph.add_node(TaskNode.from_dict(node_data))

        for edge in self._edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph._edges.append(Edge.from_dict(edge.to_dict()))
                subgraph._adjacency[edge.source_id].add(edge.target_id)
                subgraph._reverse_adjacency[edge.target_id].add(edge.source_id)

        return subgraph

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self):
        return iter(self._nodes.values())
