"""
Persistent storage module using SQLite for sessions, nodes, and memory.
Also includes PostgreSQL support for production deployments.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from core.types import (
    TaskNode, TaskGraph, MemoryEntry, Session, User, AuthToken,
    TaskStatus, MemoryType, StorageError, generate_id, current_timestamp
)

logger = logging.getLogger(__name__)


class AsyncSQLiteConnection:
    """
    Async wrapper for SQLite connections using thread pool.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish connection."""
        async with self._lock:
            if self._connection is None:
                loop = asyncio.get_event_loop()
                self._connection = await loop.run_in_executor(
                    None, self._create_connection
                )
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def execute(
        self, 
        query: str, 
        params: Tuple = ()
    ) -> sqlite3.Cursor:
        """Execute a query."""
        async with self._lock:
            if self._connection is None:
                await self.connect()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._execute_sync, query, params
            )
    
    def _execute_sync(self, query: str, params: Tuple) -> sqlite3.Cursor:
        """Synchronous execute."""
        cursor = self._connection.cursor()
        cursor.execute(query, params)
        return cursor
    
    async def executemany(
        self, 
        query: str, 
        params_list: List[Tuple]
    ) -> sqlite3.Cursor:
        """Execute many queries."""
        async with self._lock:
            if self._connection is None:
                await self.connect()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._executemany_sync, query, params_list
            )
    
    def _executemany_sync(
        self, 
        query: str, 
        params_list: List[Tuple]
    ) -> sqlite3.Cursor:
        """Synchronous executemany."""
        cursor = self._connection.cursor()
        cursor.executemany(query, params_list)
        return cursor
    
    async def fetchone(self, query: str, params: Tuple = ()) -> Optional[Dict]:
        """Execute and fetch one row."""
        cursor = await self.execute(query, params)
        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(None, cursor.fetchone)
        return dict(row) if row else None
    
    async def fetchall(self, query: str, params: Tuple = ()) -> List[Dict]:
        """Execute and fetch all rows."""
        cursor = await self.execute(query, params)
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, cursor.fetchall)
        return [dict(row) for row in rows]
    
    async def commit(self) -> None:
        """Commit transaction."""
        async with self._lock:
            if self._connection:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.commit)
    
    async def close(self) -> None:
        """Close connection."""
        async with self._lock:
            if self._connection:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._connection.close)
                self._connection = None


class SQLiteStorage:
    """
    SQLite-based persistent storage for the agentic framework.
    """
    
    def __init__(self, db_path: str = "agentic_framework.db"):
        self.db_path = db_path
        self._conn = AsyncSQLiteConnection(db_path)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return
        
        await self._conn.connect()
        
        # Create tables
        await self._create_tables()
        self._initialized = True
    
    async def _create_tables(self) -> None:
        """Create all required tables."""
        
        # Users table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                roles TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata TEXT
            )
        """)
        
        # Sessions table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                messages TEXT,
                task_graph_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Tokens table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                refresh_token TEXT,
                refresh_expires_at TEXT,
                scopes TEXT,
                is_revoked INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Task graphs table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_graphs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                owner_id TEXT,
                metadata TEXT
            )
        """)
        
        # Task nodes table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_nodes (
                id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                dependencies TEXT,
                assigned_agent TEXT,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                timeout_seconds INTEGER NOT NULL DEFAULT 300,
                priority INTEGER NOT NULL DEFAULT 5,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                metadata TEXT,
                FOREIGN KEY (graph_id) REFERENCES task_graphs(id)
            )
        """)
        
        # Graph edges table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id TEXT NOT NULL,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                FOREIGN KEY (graph_id) REFERENCES task_graphs(id),
                FOREIGN KEY (from_node_id) REFERENCES task_nodes(id),
                FOREIGN KEY (to_node_id) REFERENCES task_nodes(id),
                UNIQUE(graph_id, from_node_id, to_node_id)
            )
        """)
        
        # Memory entries table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                tags TEXT
            )
        """)
        
        # Create indexes
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_graph ON task_nodes(graph_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_graph ON graph_edges(graph_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory_entries(agent_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)"
        )
        
        await self._conn.commit()
    
    # ========================================================================
    # USER OPERATIONS
    # ========================================================================
    
    async def save_user(self, user: User) -> bool:
        """Save a user to the database."""
        try:
            await self._conn.execute("""
                INSERT OR REPLACE INTO users 
                (id, username, password_hash, salt, email, roles, 
                 created_at, last_login, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id,
                user.username,
                user.password_hash,
                user.salt,
                user.email,
                json.dumps(user.roles),
                user.created_at,
                user.last_login,
                1 if user.is_active else 0,
                json.dumps(user.metadata)
            ))
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        row = await self._conn.fetchone(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        if row:
            return self._row_to_user(row)
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        row = await self._conn.fetchone(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        if row:
            return self._row_to_user(row)
        return None
    
    def _row_to_user(self, row: Dict) -> User:
        """Convert database row to User object."""
        return User(
            id=row['id'],
            username=row['username'],
            password_hash=row['password_hash'],
            salt=row['salt'],
            email=row['email'],
            roles=json.loads(row['roles']),
            created_at=row['created_at'],
            last_login=row['last_login'],
            is_active=bool(row['is_active']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    # ========================================================================
    # SESSION OPERATIONS
    # ========================================================================
    
    async def save_session(self, session: Session) -> bool:
        """Save a session to the database."""
        try:
            await self._conn.execute("""
                INSERT OR REPLACE INTO sessions
                (id, user_id, created_at, last_activity, expires_at,
                 metadata, is_active, messages, task_graph_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.user_id,
                session.created_at,
                session.last_activity,
                session.expires_at,
                json.dumps(session.metadata),
                1 if session.is_active else 0,
                json.dumps(session.messages),
                session.task_graph_id
            ))
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        row = await self._conn.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        if row:
            return self._row_to_session(row)
        return None
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user."""
        rows = await self._conn.fetchall(
            "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1",
            (user_id,)
        )
        return [self._row_to_session(row) for row in rows]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            await self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def _row_to_session(self, row: Dict) -> Session:
        """Convert database row to Session object."""
        return Session(
            id=row['id'],
            user_id=row['user_id'],
            created_at=row['created_at'],
            last_activity=row['last_activity'],
            expires_at=row['expires_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            is_active=bool(row['is_active']),
            messages=json.loads(row['messages']) if row['messages'] else [],
            task_graph_id=row['task_graph_id']
        )
    
    # ========================================================================
    # TOKEN OPERATIONS
    # ========================================================================
    
    async def save_token(self, token: AuthToken) -> bool:
        """Save a token to the database."""
        try:
            await self._conn.execute("""
                INSERT OR REPLACE INTO tokens
                (token, user_id, created_at, expires_at, refresh_token,
                 refresh_expires_at, scopes, is_revoked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                token.token,
                token.user_id,
                token.created_at,
                token.expires_at,
                token.refresh_token,
                token.refresh_expires_at,
                json.dumps(token.scopes),
                1 if token.is_revoked else 0
            ))
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving token: {e}")
            return False
    
    async def get_token(self, token: str) -> Optional[AuthToken]:
        """Get a token from the database."""
        row = await self._conn.fetchone(
            "SELECT * FROM tokens WHERE token = ?", (token,)
        )
        if row:
            return self._row_to_token(row)
        return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        try:
            await self._conn.execute(
                "UPDATE tokens SET is_revoked = 1 WHERE token = ?", (token,)
            )
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def _row_to_token(self, row: Dict) -> AuthToken:
        """Convert database row to AuthToken object."""
        return AuthToken(
            token=row['token'],
            user_id=row['user_id'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            refresh_token=row['refresh_token'],
            refresh_expires_at=row['refresh_expires_at'],
            scopes=json.loads(row['scopes']) if row['scopes'] else [],
            is_revoked=bool(row['is_revoked'])
        )
    
    # ========================================================================
    # TASK GRAPH OPERATIONS
    # ========================================================================
    
    async def save_task_graph(self, graph: TaskGraph) -> bool:
        """Save a task graph to the database."""
        try:
            # Save graph
            await self._conn.execute("""
                INSERT OR REPLACE INTO task_graphs
                (id, name, description, status, created_at, started_at,
                 completed_at, owner_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                graph.id,
                graph.name,
                graph.description,
                graph.status.value if isinstance(graph.status, TaskStatus) else graph.status,
                graph.created_at,
                graph.started_at,
                graph.completed_at,
                graph.owner_id,
                json.dumps(graph.metadata)
            ))
            
            # Delete existing nodes and edges
            await self._conn.execute(
                "DELETE FROM task_nodes WHERE graph_id = ?", (graph.id,)
            )
            await self._conn.execute(
                "DELETE FROM graph_edges WHERE graph_id = ?", (graph.id,)
            )
            
            # Save nodes
            for node in graph.nodes.values():
                await self._save_task_node(graph.id, node)
            
            # Save edges
            for from_id, to_id in graph.edges:
                await self._conn.execute("""
                    INSERT INTO graph_edges (graph_id, from_node_id, to_node_id)
                    VALUES (?, ?, ?)
                """, (graph.id, from_id, to_id))
            
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving task graph: {e}")
            return False
    
    async def _save_task_node(self, graph_id: str, node: TaskNode) -> bool:
        """Save a task node to the database."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO task_nodes
            (id, graph_id, name, description, status, dependencies,
             assigned_agent, input_data, output_data, error_message,
             retry_count, max_retries, timeout_seconds, priority,
             created_at, started_at, completed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id,
            graph_id,
            node.name,
            node.description,
            node.status.value if isinstance(node.status, TaskStatus) else node.status,
            json.dumps(node.dependencies),
            node.assigned_agent,
            json.dumps(node.input_data),
            json.dumps(node.output_data),
            node.error_message,
            node.retry_count,
            node.max_retries,
            node.timeout_seconds,
            node.priority,
            node.created_at,
            node.started_at,
            node.completed_at,
            json.dumps(node.metadata)
        ))
        return True
    
    async def get_task_graph(self, graph_id: str) -> Optional[TaskGraph]:
        """Get a task graph by ID."""
        row = await self._conn.fetchone(
            "SELECT * FROM task_graphs WHERE id = ?", (graph_id,)
        )
        if not row:
            return None
        
        # Get nodes
        node_rows = await self._conn.fetchall(
            "SELECT * FROM task_nodes WHERE graph_id = ?", (graph_id,)
        )
        nodes = {row['id']: self._row_to_task_node(row) for row in node_rows}
        
        # Get edges
        edge_rows = await self._conn.fetchall(
            "SELECT from_node_id, to_node_id FROM graph_edges WHERE graph_id = ?",
            (graph_id,)
        )
        edges = [(row['from_node_id'], row['to_node_id']) for row in edge_rows]
        
        return TaskGraph(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            nodes=nodes,
            edges=edges,
            status=TaskStatus(row['status']),
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            owner_id=row['owner_id'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    async def list_task_graphs(
        self, 
        owner_id: str = None,
        status: TaskStatus = None,
        limit: int = 100
    ) -> List[TaskGraph]:
        """List task graphs with optional filters."""
        query = "SELECT id FROM task_graphs WHERE 1=1"
        params = []
        
        if owner_id:
            query += " AND owner_id = ?"
            params.append(owner_id)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        rows = await self._conn.fetchall(query, tuple(params))
        graphs = []
        for row in rows:
            graph = await self.get_task_graph(row['id'])
            if graph:
                graphs.append(graph)
        
        return graphs
    
    async def delete_task_graph(self, graph_id: str) -> bool:
        """Delete a task graph."""
        try:
            await self._conn.execute(
                "DELETE FROM graph_edges WHERE graph_id = ?", (graph_id,)
            )
            await self._conn.execute(
                "DELETE FROM task_nodes WHERE graph_id = ?", (graph_id,)
            )
            await self._conn.execute(
                "DELETE FROM task_graphs WHERE id = ?", (graph_id,)
            )
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting task graph: {e}")
            return False
    
    def _row_to_task_node(self, row: Dict) -> TaskNode:
        """Convert database row to TaskNode object."""
        return TaskNode(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            status=TaskStatus(row['status']),
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else [],
            assigned_agent=row['assigned_agent'],
            input_data=json.loads(row['input_data']) if row['input_data'] else {},
            output_data=json.loads(row['output_data']) if row['output_data'] else {},
            error_message=row['error_message'],
            retry_count=row['retry_count'],
            max_retries=row['max_retries'],
            timeout_seconds=row['timeout_seconds'],
            priority=row['priority'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    # ========================================================================
    # MEMORY OPERATIONS
    # ========================================================================
    
    async def save_memory_entry(self, entry: MemoryEntry) -> bool:
        """Save a memory entry to the database."""
        try:
            await self._conn.execute("""
                INSERT OR REPLACE INTO memory_entries
                (id, agent_id, memory_type, content, embedding, importance,
                 access_count, last_accessed, created_at, expires_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.agent_id,
                entry.memory_type.value if isinstance(entry.memory_type, MemoryType) else entry.memory_type,
                json.dumps(entry.content),
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.importance,
                entry.access_count,
                entry.last_accessed,
                entry.created_at,
                entry.expires_at,
                json.dumps(entry.tags)
            ))
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving memory entry: {e}")
            return False
    
    async def get_memory_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        row = await self._conn.fetchone(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        )
        if row:
            return self._row_to_memory_entry(row)
        return None
    
    async def get_agent_memories(
        self,
        agent_id: str,
        memory_type: MemoryType = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get memories for an agent."""
        query = "SELECT * FROM memory_entries WHERE agent_id = ?"
        params = [agent_id]
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        rows = await self._conn.fetchall(query, tuple(params))
        return [self._row_to_memory_entry(row) for row in rows]
    
    async def delete_memory_entry(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            await self._conn.execute(
                "DELETE FROM memory_entries WHERE id = ?", (entry_id,)
            )
            await self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting memory entry: {e}")
            return False
    
    async def cleanup_expired_memories(self) -> int:
        """Delete expired memory entries."""
        try:
            now = current_timestamp()
            cursor = await self._conn.execute("""
                DELETE FROM memory_entries 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (now,))
            await self._conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up expired memories: {e}")
            return 0
    
    def _row_to_memory_entry(self, row: Dict) -> MemoryEntry:
        """Convert database row to MemoryEntry object."""
        return MemoryEntry(
            id=row['id'],
            agent_id=row['agent_id'],
            memory_type=MemoryType(row['memory_type']),
            content=json.loads(row['content']),
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            tags=json.loads(row['tags']) if row['tags'] else []
        )
    
    # ========================================================================
    # GENERIC OPERATIONS
    # ========================================================================
    
    async def save(
        self, 
        collection: str, 
        key: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Generic save operation for arbitrary data."""
        # Create collection table if not exists
        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {collection} (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        now = current_timestamp()
        await self._conn.execute(f"""
            INSERT OR REPLACE INTO {collection} (key, data, created_at, updated_at)
            VALUES (?, ?, COALESCE(
                (SELECT created_at FROM {collection} WHERE key = ?), ?
            ), ?)
        """, (key, json.dumps(data), key, now, now))
        await self._conn.commit()
        return True
    
    async def load(
        self, 
        collection: str, 
        key: str
    ) -> Optional[Dict[str, Any]]:
        """Generic load operation."""
        try:
            row = await self._conn.fetchone(
                f"SELECT data FROM {collection} WHERE key = ?", (key,)
            )
            if row:
                return json.loads(row['data'])
        except:
            pass
        return None
    
    async def delete(self, collection: str, key: str) -> bool:
        """Generic delete operation."""
        try:
            await self._conn.execute(
                f"DELETE FROM {collection} WHERE key = ?", (key,)
            )
            await self._conn.commit()
            return True
        except:
            return False
    
    async def query(
        self, 
        collection: str, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generic query operation with filters."""
        try:
            rows = await self._conn.fetchall(
                f"SELECT data FROM {collection}"
            )
            results = []
            for row in rows:
                data = json.loads(row['data'])
                # Apply filters
                match = True
                for key, value in filters.items():
                    if data.get(key) != value:
                        match = False
                        break
                if match:
                    results.append(data)
            return results
        except:
            return []
    
    async def close(self) -> None:
        """Close the database connection."""
        await self._conn.close()