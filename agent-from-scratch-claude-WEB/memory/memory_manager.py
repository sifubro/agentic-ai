"""
Multi-agent memory module with short-term, long-term, episodic, 
semantic, and procedural memory support.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple,
    Union, TYPE_CHECKING
)
import logging
import json

from core.types import (
    MemoryEntry, MemoryType, generate_id, current_timestamp, 
    is_expired, parse_timestamp
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple in-memory vector store for semantic search.
    Uses cosine similarity for nearest neighbor search.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def add(
        self, 
        id: str, 
        vector: List[float],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add a vector to the store."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        self._vectors[id] = vector
        self._metadata[id] = metadata or {}
    
    async def remove(self, id: str) -> bool:
        """Remove a vector from the store."""
        if id in self._vectors:
            del self._vectors[id]
            del self._metadata[id]
            return True
        return False
    
    async def search(
        self, 
        query_vector: List[float],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if not self._vectors:
            return []
        
        results = []
        for id, vector in self._vectors.items():
            similarity = self.cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                results.append((id, similarity, self._metadata[id]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a vector and its metadata by ID."""
        if id in self._vectors:
            return self._vectors[id], self._metadata[id]
        return None
    
    def __len__(self) -> int:
        return len(self._vectors)


class SimpleEmbedding:
    """Simple text embedding using character n-grams."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def embed(self, text: str) -> List[float]:
        """Generate a simple embedding for text."""
        text = text.lower().strip()
        
        ngrams: List[str] = []
        for n in [2, 3, 4]:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        
        vector = [0.0] * self.dimension
        for ngram in ngrams:
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            pos = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vector[pos] += val
        
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [await self.embed(text) for text in texts]


class MemoryStore:
    """Memory store for a single agent with different memory types."""
    
    def __init__(
        self,
        agent_id: str,
        embedding_dimension: int = 384,
        short_term_capacity: int = 100,
        short_term_ttl_seconds: int = 3600
    ):
        self.agent_id = agent_id
        self.embedding_dimension = embedding_dimension
        self.short_term_capacity = short_term_capacity
        self.short_term_ttl_seconds = short_term_ttl_seconds
        
        self._memories: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            mem_type: {} for mem_type in MemoryType
        }
        
        self._vector_store = VectorStore(embedding_dimension)
        self._embedder = SimpleEmbedding(embedding_dimension)
        self._short_term_heap: List[Tuple[float, str]] = []
        self._lock = asyncio.Lock()
    
    async def store(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        tags: List[str] = None,
        expires_in_seconds: int = None
    ) -> MemoryEntry:
        """Store a memory entry."""
        async with self._lock:
            entry = MemoryEntry(
                id=generate_id(),
                agent_id=self.agent_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                tags=tags or [],
                created_at=current_timestamp(),
                last_accessed=current_timestamp()
            )
            
            if expires_in_seconds is not None:
                entry.expires_at = (
                    datetime.utcnow() + timedelta(seconds=expires_in_seconds)
                ).isoformat()
            elif memory_type == MemoryType.SHORT_TERM:
                entry.expires_at = (
                    datetime.utcnow() + timedelta(seconds=self.short_term_ttl_seconds)
                ).isoformat()
            
            text_content = self._extract_text(content)
            if text_content:
                entry.embedding = await self._embedder.embed(text_content)
                await self._vector_store.add(
                    entry.id,
                    entry.embedding,
                    {"memory_type": memory_type.value, "importance": importance}
                )
            
            self._memories[memory_type][entry.id] = entry
            
            if memory_type == MemoryType.SHORT_TERM:
                heapq.heappush(self._short_term_heap, (importance, entry.id))
                await self._enforce_short_term_capacity()
            
            return entry
    
    async def retrieve(
        self,
        memory_id: str,
        memory_type: MemoryType = None
    ) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        async with self._lock:
            if memory_type:
                entry = self._memories[memory_type].get(memory_id)
            else:
                for mem_store in self._memories.values():
                    if memory_id in mem_store:
                        entry = mem_store[memory_id]
                        break
                else:
                    return None
            
            if entry:
                entry.access_count += 1
                entry.last_accessed = current_timestamp()
            
            return entry
    
    async def search_semantic(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[MemoryEntry]:
        """Search memories using semantic similarity."""
        query_embedding = await self._embedder.embed(query)
        results = await self._vector_store.search(
            query_embedding, top_k=top_k * 2, threshold=threshold
        )
        
        entries = []
        for memory_id, score, metadata in results:
            entry = await self.retrieve(memory_id)
            if entry:
                if memory_types is None or entry.memory_type in memory_types:
                    entries.append(entry)
                    if len(entries) >= top_k:
                        break
        
        return entries
    
    async def search_by_tags(
        self,
        tags: List[str],
        memory_types: List[MemoryType] = None,
        match_all: bool = False
    ) -> List[MemoryEntry]:
        """Search memories by tags."""
        results = []
        search_types = memory_types or list(MemoryType)
        
        for mem_type in search_types:
            for entry in self._memories[mem_type].values():
                if match_all:
                    if all(tag in entry.tags for tag in tags):
                        results.append(entry)
                else:
                    if any(tag in entry.tags for tag in tags):
                        results.append(entry)
        
        results.sort(key=lambda e: e.importance, reverse=True)
        return results
    
    async def search_recent(
        self,
        memory_types: List[MemoryType] = None,
        limit: int = 10,
        since: datetime = None
    ) -> List[MemoryEntry]:
        """Get recent memories."""
        results = []
        search_types = memory_types or list(MemoryType)
        
        for mem_type in search_types:
            for entry in self._memories[mem_type].values():
                if since:
                    created = parse_timestamp(entry.created_at)
                    if created < since:
                        continue
                results.append(entry)
        
        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]
    
    async def forget(
        self,
        memory_id: str,
        memory_type: MemoryType = None
    ) -> bool:
        """Remove a memory."""
        async with self._lock:
            if memory_type:
                if memory_id in self._memories[memory_type]:
                    del self._memories[memory_type][memory_id]
                    await self._vector_store.remove(memory_id)
                    return True
            else:
                for mem_store in self._memories.values():
                    if memory_id in mem_store:
                        del mem_store[memory_id]
                        await self._vector_store.remove(memory_id)
                        return True
            return False
    
    async def consolidate(
        self,
        source_type: MemoryType,
        target_type: MemoryType,
        importance_threshold: float = 0.7
    ) -> int:
        """Consolidate memories from one type to another."""
        async with self._lock:
            consolidated = 0
            to_move = []
            
            for entry in self._memories[source_type].values():
                if entry.importance >= importance_threshold:
                    to_move.append(entry)
            
            for entry in to_move:
                del self._memories[source_type][entry.id]
                entry.memory_type = target_type
                entry.expires_at = None
                self._memories[target_type][entry.id] = entry
                consolidated += 1
            
            return consolidated
    
    async def decay(self, decay_rate: float = 0.1) -> int:
        """Apply memory decay."""
        async with self._lock:
            decayed = 0
            now = datetime.utcnow()
            
            for mem_type in [MemoryType.SHORT_TERM, MemoryType.EPISODIC]:
                to_remove = []
                
                for entry in self._memories[mem_type].values():
                    last_accessed = parse_timestamp(entry.last_accessed)
                    hours_since_access = (now - last_accessed).total_seconds() / 3600
                    decay_factor = decay_rate * hours_since_access / (1 + entry.access_count * 0.1)
                    entry.importance = max(0.0, entry.importance - decay_factor)
                    decayed += 1
                    
                    if entry.importance < 0.01:
                        to_remove.append(entry.id)
                
                for memory_id in to_remove:
                    del self._memories[mem_type][memory_id]
                    await self._vector_store.remove(memory_id)
            
            return decayed
    
    async def cleanup_expired(self) -> int:
        """Remove expired memories."""
        async with self._lock:
            removed = 0
            
            for mem_type, mem_store in self._memories.items():
                to_remove = []
                for memory_id, entry in mem_store.items():
                    if entry.expires_at and is_expired(entry.expires_at):
                        to_remove.append(memory_id)
                
                for memory_id in to_remove:
                    del mem_store[memory_id]
                    await self._vector_store.remove(memory_id)
                    removed += 1
            
            return removed
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "agent_id": self.agent_id,
            "total_memories": sum(len(m) for m in self._memories.values()),
            "by_type": {
                mem_type.value: len(mem_store)
                for mem_type, mem_store in self._memories.items()
            },
            "vector_store_size": len(self._vector_store)
        }
    
    def _extract_text(self, content: Dict[str, Any]) -> str:
        """Extract text from content for embedding."""
        parts = []
        
        for key, value in content.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
            elif isinstance(value, dict):
                parts.append(self._extract_text(value))
        
        return " ".join(parts)
    
    async def _enforce_short_term_capacity(self):
        """Remove lowest importance memories if over capacity."""
        while len(self._memories[MemoryType.SHORT_TERM]) > self.short_term_capacity:
            if not self._short_term_heap:
                break
            _, memory_id = heapq.heappop(self._short_term_heap)
            if memory_id in self._memories[MemoryType.SHORT_TERM]:
                del self._memories[MemoryType.SHORT_TERM][memory_id]
                await self._vector_store.remove(memory_id)


class MultiAgentMemoryManager:
    """Manages memory across multiple agents with shared memory support."""
    
    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
        self._agent_stores: Dict[str, MemoryStore] = {}
        self._shared_memories: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def get_or_create_store(self, agent_id: str) -> MemoryStore:
        """Get or create a memory store for an agent."""
        async with self._lock:
            if agent_id not in self._agent_stores:
                self._agent_stores[agent_id] = MemoryStore(
                    agent_id=agent_id,
                    embedding_dimension=self.embedding_dimension
                )
            return self._agent_stores[agent_id]
    
    async def share_memory(
        self,
        memory_id: str,
        from_agent_id: str,
        to_agent_ids: List[str]
    ) -> bool:
        """Share a memory from one agent to others."""
        source_store = await self.get_or_create_store(from_agent_id)
        entry = await source_store.retrieve(memory_id)
        
        if not entry:
            return False
        
        for agent_id in to_agent_ids:
            target_store = await self.get_or_create_store(agent_id)
            await target_store.store(
                content=entry.content.copy(),
                memory_type=entry.memory_type,
                importance=entry.importance,
                tags=entry.tags + [f"shared_from:{from_agent_id}"]
            )
            self._shared_memories[memory_id].add(agent_id)
        
        return True
    
    async def broadcast_memory(
        self,
        content: Dict[str, Any],
        from_agent_id: str,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> List[str]:
        """Broadcast a memory to all agents."""
        memory_ids = []
        
        for agent_id, store in self._agent_stores.items():
            entry = await store.store(
                content=content.copy(),
                memory_type=MemoryType.SHORT_TERM,
                importance=importance,
                tags=(tags or []) + [f"broadcast_from:{from_agent_id}"]
            )
            memory_ids.append(entry.id)
        
        return memory_ids
    
    async def collective_search(
        self,
        query: str,
        agent_ids: List[str] = None,
        top_k: int = 10
    ) -> Dict[str, List[MemoryEntry]]:
        """Search across multiple agent memory stores."""
        results = {}
        search_agents = agent_ids or list(self._agent_stores.keys())
        
        for agent_id in search_agents:
            if agent_id in self._agent_stores:
                store = self._agent_stores[agent_id]
                entries = await store.search_semantic(query, top_k=top_k)
                if entries:
                    results[agent_id] = entries
        
        return results
    
    async def transfer_memory(
        self,
        memory_id: str,
        from_agent_id: str,
        to_agent_id: str
    ) -> bool:
        """Transfer a memory from one agent to another."""
        source_store = self._agent_stores.get(from_agent_id)
        if not source_store:
            return False
        
        entry = await source_store.retrieve(memory_id)
        if not entry:
            return False
        
        target_store = await self.get_or_create_store(to_agent_id)
        await target_store.store(
            content=entry.content.copy(),
            memory_type=entry.memory_type,
            importance=entry.importance,
            tags=entry.tags + [f"transferred_from:{from_agent_id}"]
        )
        
        await source_store.forget(memory_id)
        return True
    
    async def get_collective_stats(self) -> Dict[str, Any]:
        """Get statistics across all agents."""
        stats = {
            "total_agents": len(self._agent_stores),
            "total_memories": 0,
            "shared_memories": len(self._shared_memories),
            "by_agent": {}
        }
        
        for agent_id, store in self._agent_stores.items():
            agent_stats = await store.get_stats()
            stats["by_agent"][agent_id] = agent_stats
            stats["total_memories"] += agent_stats["total_memories"]
        
        return stats
    
    async def cleanup_all(self) -> Dict[str, int]:
        """Run cleanup on all agent stores."""
        results = {}
        for agent_id, store in self._agent_stores.items():
            expired = await store.cleanup_expired()
            decayed = await store.decay()
            results[agent_id] = {"expired": expired, "decayed": decayed}
        return results


class PlannerMemory:
    """Specialized memory for planner agents."""
    
    def __init__(self, agent_id: str, memory_manager: MultiAgentMemoryManager):
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        self._plans: Dict[str, Dict[str, Any]] = {}
        self._goals: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
    
    async def store_plan(
        self,
        plan_id: str,
        plan: Dict[str, Any],
        goals: List[str]
    ) -> str:
        """Store a plan with associated goals."""
        self._plans[plan_id] = {
            "plan": plan,
            "goals": goals,
            "created_at": current_timestamp(),
            "status": "pending"
        }
        
        store = await self.memory_manager.get_or_create_store(self.agent_id)
        await store.store(
            content={
                "type": "plan",
                "plan_id": plan_id,
                "goals": goals,
                "summary": plan.get("summary", "")
            },
            memory_type=MemoryType.PROCEDURAL,
            importance=0.8,
            tags=["plan"] + goals
        )
        
        return plan_id
    
    async def update_plan_status(
        self,
        plan_id: str,
        status: str,
        result: Dict[str, Any] = None
    ):
        """Update plan execution status."""
        if plan_id in self._plans:
            self._plans[plan_id]["status"] = status
            self._plans[plan_id]["updated_at"] = current_timestamp()
            if result:
                self._plans[plan_id]["result"] = result
    
    async def record_execution(
        self,
        plan_id: str,
        step: str,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Record execution step."""
        record = {
            "plan_id": plan_id,
            "step": step,
            "success": success,
            "details": details or {},
            "timestamp": current_timestamp()
        }
        self._execution_history.append(record)
        
        store = await self.memory_manager.get_or_create_store(self.agent_id)
        await store.store(
            content=record,
            memory_type=MemoryType.EPISODIC,
            importance=0.6 if success else 0.8,
            tags=["execution", "success" if success else "failure"]
        )
    
    async def get_similar_plans(
        self,
        goal_description: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar past plans for a goal."""
        store = await self.memory_manager.get_or_create_store(self.agent_id)
        entries = await store.search_semantic(
            goal_description,
            memory_types=[MemoryType.PROCEDURAL],
            top_k=top_k
        )
        
        plans = []
        for entry in entries:
            plan_id = entry.content.get("plan_id")
            if plan_id and plan_id in self._plans:
                plans.append(self._plans[plan_id])
        
        return plans
    
    async def get_failure_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failure patterns from execution history."""
        failures = [
            record for record in self._execution_history
            if not record["success"]
        ]
        return failures[-limit:]
    
    async def learn_from_failure(
        self,
        plan_id: str,
        failure_reason: str,
        correction: Dict[str, Any]
    ):
        """Store learning from a failure."""
        store = await self.memory_manager.get_or_create_store(self.agent_id)
        await store.store(
            content={
                "type": "learning",
                "plan_id": plan_id,
                "failure_reason": failure_reason,
                "correction": correction
            },
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            tags=["learning", "failure_correction"]
        )