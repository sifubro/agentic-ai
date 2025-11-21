"""
Advanced Agentic Orchestrator

Features added on top of the original implementation:
 a) concurrency via asyncio (parallel execution of independent nodes)
 b) cycle detection on graph insertion (raises on cycles)
 c) PBKDF2 password hashing and token refresh/revocation
 d) minimal HTTP API (no external frameworks) using built-in http.server and threading
 e) persistence using SQLite for users, tokens, sessions, nodes, and long-term memory
 f) unit tests and type annotations

Notes:
- This file is a single-file demo/prototype. It's still educational and not production-secure.
- The HTTP server exposes limited endpoints for demonstration: /create_user, /login, /refresh, /revoke, /create_session, /execute_graph, /session_status, /recall
- The orchestrator runs nodes concurrently using asyncio tasks and limits concurrency via a semaphore.

Run: python3 advanced_agentic_orchestrator.py

"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import secrets
import hashlib
import hmac
import base64
import logging
import uuid
import threading
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import unittest

# ---------- Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedOrchestrator")

# ---------- Utilities & Types

def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ensure_json_serializable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return {k: ensure_json_serializable(v) for k, v in obj.__dict__.items()}
        return str(obj)

JSONDict = Dict[str, Any]

# ---------- Persistence (SQLite)

DB_PATH = os.environ.get("AGENT_DB_PATH", "agentic_orchestrator.db")

class Persistence:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        init_needed = not os.path.exists(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if init_needed:
            self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT UNIQUE, pw_hash TEXT, salt TEXT, created_at REAL);
        CREATE TABLE tokens (token TEXT PRIMARY KEY, user_id TEXT, expires_at REAL, refresh_token TEXT);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, user_id TEXT, created_at REAL, expires_at REAL, metadata TEXT);
        CREATE TABLE nodes (node_id TEXT PRIMARY KEY, name TEXT, agent_id TEXT, action TEXT, inputs TEXT, metadata TEXT, status TEXT, attempts INTEGER, result TEXT);
        CREATE TABLE memories (mem_id TEXT PRIMARY KEY, user_id TEXT, ts REAL, item TEXT);
        """)
        self.conn.commit()
        logger.info("Initialized SQLite DB at %s", self.path)

    # User methods
    def save_user(self, user_id: str, username: str, pw_hash: str, salt: str) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO users (user_id, username, pw_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, username, pw_hash, salt, now_ts()))
        self.conn.commit()

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()

    # Token methods
    def save_token(self, token: str, user_id: str, expires_at: float, refresh_token: Optional[str] = None) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO tokens (token, user_id, expires_at, refresh_token) VALUES (?, ?, ?, ?)", (token, user_id, expires_at, refresh_token))
        self.conn.commit()

    def get_token(self, token: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM tokens WHERE token = ?", (token,))
        return cur.fetchone()

    def delete_token(self, token: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM tokens WHERE token = ?", (token,))
        self.conn.commit()

    def update_token_expiry(self, token: str, new_expiry: float) -> None:
        cur = self.conn.cursor()
        cur.execute("UPDATE tokens SET expires_at = ? WHERE token = ?", (new_expiry, token))
        self.conn.commit()

    # Session methods
    def save_session(self, session_id: str, user_id: str, created_at: float, expires_at: float, metadata: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO sessions (session_id, user_id, created_at, expires_at, metadata) VALUES (?, ?, ?, ?, ?)",
                    (session_id, user_id, created_at, expires_at, json.dumps(metadata)))
        self.conn.commit()

    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return cur.fetchone()

    def delete_session(self, session_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def update_session_expiry(self, session_id: str, new_expiry: float) -> None:
        cur = self.conn.cursor()
        cur.execute("UPDATE sessions SET expires_at = ? WHERE session_id = ?", (new_expiry, session_id))
        self.conn.commit()

    # Nodes (persisting node state)
    def save_node(self, node: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO nodes (node_id, name, agent_id, action, inputs, metadata, status, attempts, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (node["node_id"], node["name"], node["agent_id"], node["action"], json.dumps(node.get("inputs", [])), json.dumps(node.get("metadata", {})), node.get("status"), node.get("attempts", 0), json.dumps(node.get("result"))))
        self.conn.commit()

    def load_all_nodes(self) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes")
        return cur.fetchall()

    # Memories
    def add_memory(self, mem_id: str, user_id: str, ts: float, item: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memories (mem_id, user_id, ts, item) VALUES (?, ?, ?, ?)", (mem_id, user_id, ts, json.dumps(item)))
        self.conn.commit()

    def get_memories_for_user(self, user_id: str) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memories WHERE user_id = ? ORDER BY ts DESC", (user_id,))
        return cur.fetchall()

# ---------- Authentication Manager (PBKDF2 + tokens + refresh)

class AuthError(Exception):
    pass

class AuthManager:
    def __init__(self, persistence: Persistence, token_ttl_seconds: int = 3600, refresh_ttl_seconds: int = 86400):
        self._p = persistence
        self._token_ttl = token_ttl_seconds
        self._refresh_ttl = refresh_ttl_seconds
        logger.info("AuthManager ready (token ttl=%ds, refresh ttl=%ds)", token_ttl_seconds, refresh_ttl_seconds)

    @staticmethod
    def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 100_000) -> str:
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)
        return base64.b64encode(dk).decode()

    def create_user(self, username: str, password: str) -> str:
        if self._p.get_user(username) is not None:
            raise AuthError("username_already_exists")
        user_id = generate_id("user")
        salt = secrets.token_bytes(16)
        pw_hash = self._pbkdf2_hash(password, salt)
        # store salt as base64
        self._p.save_user(user_id, username, pw_hash, base64.b64encode(salt).decode())
        logger.info("Created user %s (%s)", username, user_id)
        return user_id

    def login(self, username: str, password: str) -> Tuple[str, str]:
        '''
        When a user first logs in with a username and password, your system creates two tokens:
        
        Access Token: Proves the user is authenticated when calling APIs. Short (e.g., 1 hour)

        Refresh Token: Lets the user get a new access token when the old one expires, without re-entering the password. Long (e.g., 1 day or 1 week)

        So the password is needed only once — at the start of the session. After that, the refresh token stands in for the password for a while.
        '''
        # login() → first time token creation (user logs in with username & password).
        row = self._p.get_user(username)
        if not row:
            raise AuthError("invalid_credentials")
        # a cryptographically random 16-byte value, unique per user.
        salt = base64.b64decode(row["salt"])
        expected = row["pw_hash"]
        actual = self._pbkdf2_hash(password, salt) #That produces a derived key (a binary digest) that is then base64 encoded and stored as pw_hash
        if not hmac.compare_digest(expected, actual):
            raise AuthError("invalid_credentials")
        # this is your access token (short-lived, used for API calls).
        token = secrets.token_urlsafe(32)
        expires_at = now_ts() + self._token_ttl
        # long-lived token to renew the access token later.
        refresh_token = secrets.token_urlsafe(32)
        # store token
        self._p.save_token(token, row["user_id"], expires_at, refresh_token)
        logger.info("User %s logged in; token=%s expires=%s", username, token, expires_at)
        return token, refresh_token

    def validate_token(self, token: str) -> Optional[str]:
        row = self._p.get_token(token)
        if not row:
            return None
        if now_ts() > row["expires_at"]:
            # Token expired; delete
            self._p.delete_token(token)
            return None
        return row["user_id"]

    def refresh_token(self, refresh_token: str) -> Tuple[str, str]:
        '''
        This method is called when a client presents a refresh token and asks for a new access token.
        refresh_token() → subsequent renewals (user logs in without password, using refresh token)

        Refresh Token: Used only to obtain a new access token when the old one expires. Lifetime: Long (e.g., 1 day). Kept secret by the client
        Store the new refresh token for the next renewal cycle.
        Lets the user get a new access token when the old one expires, without re-entering the password.
        
        Access Token: Used to authenticate API calls (e.g., “user X is allowed to do Y”). Lifetime: Short (e.g., 1 hour). Sent with each request
        Use the new access token to make API calls.
        
        
        when your access token expires, you send your refresh token to the server to get a new one.
        '''
        #Looks up the given refresh_token in the database.
        # Finds which user_id it belongs to.
        #Creates a new access token and new refresh token.
        #Deletes the old one (so it can’t be reused).
        #Returns the new pair: { "token": "ACCESS789", "refresh_token": "REFRESH999" }

        #So the user didn’t need to type the password again, but the system still verified identity — because the
        # refresh token is cryptographically random, stored securely, and bound to that user in the database.
        #Refresh tokens must be stored very securely (e.g., in an encrypted cookie, secure storage on mobile).


        # find token row with this refresh_token
        cur = self._p.conn.cursor()
        cur.execute("SELECT token, user_id FROM tokens WHERE refresh_token = ?", (refresh_token,))
        row = cur.fetchone()
        if not row:
            raise AuthError("invalid_refresh")
        # issue new token and refresh token
        new_token = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)
        new_expiry = now_ts() + self._token_ttl
        # delete old token row
        # This revokes the old access/refresh token pair.
        # That means each refresh token can be used only once — a good security practice (prevents reuse if someone steals it).
        cur.execute("DELETE FROM tokens WHERE refresh_token = ?", (refresh_token,))
        cur.execute("INSERT INTO tokens (token, user_id, expires_at, refresh_token) VALUES (?, ?, ?, ?)", (new_token, row["user_id"], new_expiry, new_refresh))
        self._p.conn.commit()
        
        return new_token, new_refresh

    def revoke_token(self, token: str) -> None:
        self._p.delete_token(token)

# ---------- Session Manager (backed by SQLite)

class SessionExpired(Exception):
    pass

class SessionManager:
    def __init__(self, persistence: Persistence, session_ttl_seconds: int = 1800):
        self._p = persistence
        self._session_ttl = session_ttl_seconds
        logger.info("SessionManager ready (ttl=%ds)", session_ttl_seconds)

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = generate_id("sess")
        now = now_ts()
        expires = now + self._session_ttl
        self._p.save_session(session_id, user_id, now, expires, metadata or {})
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        row = self._p.get_session(session_id)
        if not row:
            raise SessionExpired("session_not_found")
        if now_ts() > row["expires_at"]:
            self._p.delete_session(session_id)
            raise SessionExpired("session_expired")
        return {"session_id": row["session_id"], "user_id": row["user_id"], "created_at": row["created_at"], "expires_at": row["expires_at"], "metadata": json.loads(row["metadata"]) if row["metadata"] else {}}

    def touch_session(self, session_id: str) -> None:
        new_expiry = now_ts() + self._session_ttl
        self._p.update_session_expiry(session_id, new_expiry)

    def end_session(self, session_id: str) -> None:
        self._p.delete_session(session_id)

# ---------- Memory Manager (uses SQLite long-term + in-memory short-term)

class MemoryManager:
    def __init__(self, persistence: Persistence, short_term_max: int = 100):
        self._p = persistence
        self._short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=short_term_max))

    def add_short(self, session_id: str, item: Dict[str, Any]) -> None:
        self._short_term[session_id].append({"ts": now_ts(), "item": ensure_json_serializable(item)})

    def list_short(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._short_term.get(session_id, []))

    def add_long(self, user_id: str, item: Dict[str, Any]) -> None:
        mem_id = generate_id("mem")
        self._p.add_memory(mem_id, user_id, now_ts(), ensure_json_serializable(item))

    def recall(self, user_id: str, keywords: List[str], session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        results = []
        def match(item):
            text = json.dumps(item, ensure_ascii=False).lower()
            return any(kw.lower() in text for kw in keywords)

        if session_id:
            for rec in reversed(self._short_term.get(session_id, [])):
                if match(rec["item"]):
                    results.append(rec["item"])
                    if len(results) >= limit:
                        return results
        rows = self._p.get_memories_for_user(user_id)
        for r in rows:
            item = json.loads(r["item"])
            if match(item):
                results.append(item)
                if len(results) >= limit:
                    break
        return results

# ---------- Messaging

class MessageValidationError(Exception):
    pass

class Message:
    '''
    A Message is a standardized envelope that carries data between components of your distributed AI system — like emails between people, but inside your orchestration graph.
    In this architecture, everything talks via messages:
    - Agents send messages to nodes.
    - Nodes send messages to other nodes.
    - Orchestrators route messages between agents.

    A Message is the universal communication envelope used for:
    - Sending a task from the GraphOrchestrator to an Agent.
    - Potentially passing data between TaskNodes or agents.
    - Carrying contextual metadata for execution and debugging.

    Think of it as a miniature "packet" that moves through the workflow graph — every task runs because it received a Message.

    **self.frm** = Who sent the message (the sender). Usually, this is the session ID (session["session_id"]), 
    meaning the user session or orchestrator initiated this message.
    Could also be an agent ID if one agent sends a message to another (in more advanced designs).
    Logging and tracing who requested a task. Attribution and access control (via AuthManager).
    frm = "sess_001"  # the session that initiated the task
    
    **self.to** = The recipient — which agent should handle this message. 
    The orchestrator sets this field automatically from the node definition:
    e.g. msg = Message(
            frm=session["session_id"],
            to=node.agent_id,
            task_id=node.node_id,
            payload=...,
        )
    Used by The orchestrator to route the task to the correct agent (orchestrator.get_agent(msg.to)).
    Used by The agent to know that it is the one responsible for this task.

    **self.task_id** = Identify the job “This is job #42.”  (identity)
    node_id = task_id in messages
    A string (e.g. "node_002") that uniquely identifies a TaskNode in the graph.
    Each node in the workflow graph gets its own node_id (which becomes task_id in messages).
    Unique ID of the TaskNode this message corresponds to. It connects the message to a specific node in the workflow graph
    Why it exists:
        - To know which step the agent is performing.
        - To store the results in the correct node.
        - To track dependencies (parent → child relationships).
    This is essential for: 
        - Mapping results back to the correct node. 
        - Saving task outcomes to the persistence layer. 
        - Linking dependencies (inputs).
    e.g. task_id = "node_002"  # corresponds to the “Research Topic” TaskNode


    **self.payload** = Describe what to do “Here’s what I want you to do and with what data.”  (content)
    The core content — what the agent actually needs to process.
    It usually contains: 
        - The action to perform (e.g., "research", "summarize", etc.)
        - The inputs (data from parent nodes).
        - Any metadata (parameters, topic, etc.) needed for execution.
    Why it exists:
        - The agent doesn’t know about graphs or sessions. It just receives a self-contained task description — the payload.
        - It’s how the orchestrator hands off computation from one node to the next.


    **self.metadata** = Provide context / tracing info “FYI, this came from Node A in Workflow B.”  (context)
    Lightweight information that helps describe how or why the message exists — not the main data, but useful context.
    Examples include: Node name, Agent action, "skill", Task priority or "deadline", Debug trace info, "retries" or "run_mode" 

    self.ts = When the message was created. Automatically set via now_ts().
    Used for: Ordering messages in time. Logging and debugging delays. Possibly timeouts or tracing.

    '''
    REQUIRED_KEYS = {"from", "to", "task_id", "payload"}

    def __init__(self, frm: str, to: str, task_id: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.frm = frm
        self.to = to
        self.task_id = task_id
        self.payload = payload
        self.metadata = metadata or {}
        self.ts = now_ts()

    def to_dict(self) -> Dict[str, Any]:
        return {"from": self.frm, "to": self.to, "task_id": self.task_id, "payload": ensure_json_serializable(self.payload), "metadata": ensure_json_serializable(self.metadata), "ts": self.ts}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def parse(raw: Any) -> "Message":
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raise MessageValidationError("invalid_json")
        if not isinstance(raw, dict):
            raise MessageValidationError("message_must_be_object")
        missing = Message.REQUIRED_KEYS - set(raw.keys())
        if missing:
            raise MessageValidationError(f"missing_keys: {sorted(list(missing))}")
        return Message(frm=str(raw["from"]), to=str(raw["to"]), task_id=str(raw["task_id"]), payload=raw["payload"], metadata=raw.get("metadata"))


def validate_agent_response(resp: Dict[str, Any]) -> None:
    if not isinstance(resp, dict):
        raise MessageValidationError("response_must_be_object")
    if "status" not in resp:
        raise MessageValidationError("response_missing_status")
    if resp["status"] not in {"ok", "error"}:
        raise MessageValidationError("response_status_invalid")
    if "next_tasks" in resp and not isinstance(resp["next_tasks"], list):
        raise MessageValidationError("next_tasks_must_be_list")

# ---------- Agent

class Agent:
    def __init__(self, agent_id: str, name: str, role: str = "assistant") -> None:
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self._skills: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Union[Dict[str, Any], asyncio.Future]]] = {}

    def register_skill(self, name: str, func: Callable[[Dict[str, Any], Dict[str, Any]], Union[Dict[str, Any], asyncio.Future]]) -> None:
        '''
        Use this to “teach” an agent what it can do.
        Each skill is a function that takes two parameters:
        - payload: the message payload (the data of the request)
        - ctx: the context (session info, agent info, memory handle, etc.)
        The skill returns a dictionary result (sync or async).
        '''
        self._skills[name] = func

    async def handle_message(self, message: Message, session: Dict[str, Any], memory: MemoryManager) -> Dict[str, Any]:
        '''
        This method is called whenever the agent receives a message (task request).

        Understanding the ctx (Context Object): This context is the glue between the agent, the system, and the session.
        It gives every skill function access to important runtime information.

        "agent": Info about the current agent (id, name, role). For logging, attribution, and when multiple agents collaborate


        '''
        skill = message.metadata.get("skill") or message.payload.get("action") or "default"
        if skill not in self._skills:
            return {"status": "error", "error": f"missing_skill:{skill}"}
        ctx = {"agent": {"id": self.agent_id, "name": self.name, "role": self.role}, 
               "session": session, 
               "message_meta": message.metadata, 
               "ts": now_ts(), 
               "memory": memory
               }
        func = self._skills[skill]
        try:
            maybe_coro = func(message.payload, ctx)
            if asyncio.iscoroutine(maybe_coro):
                result = await maybe_coro
            else:
                result = maybe_coro
            if not isinstance(result, dict):
                return {"status": "error", "error": "skill_returned_non_object"}
            validate_agent_response(result)
            return result
        except Exception as e:
            logger.exception("Agent skill errored")
            return {"status": "error", "error": f"skill_exception:{str(e)}"}

# ---------- Task Node

class TaskNode:
    '''
    It represents a single unit of work inside the orchestration graph managed by the GraphOrchestrator
    Think of a TaskNode as A task that an agent must perform, which may depend on outputs from other tasks.
    
    Each node holds:
    - Identity — a unique ID for referencing it in the graph.
    - Assignment — which agent should handle it.
    - Behavior — what action (skill) the agent should perform.
    - Dependencies — other nodes whose results it needs first.
    - Metadata — additional contextual information.
    - State — whether it’s pending, running, done, or failed.

    node_id = Unique ID for the task in the graph. Used for dependencies (inputs), logging, persistence, and reloading e.g. "node1", "summarize_task_23"
    Required for persistence and dependency tracking. Other nodes refer to this one via their inputs. If you dynamically add nodes at runtime, IDs prevent collisions.

    name = Human-readable label for debugging or UI display e.g. "Summarize Article"
    Helps you understand what each node does when visualizing a graph e.g. name="Load and Clean Dataset"

    agent_id = The agent responsible for executing the node (links to an Agent registered in the orchestrator) e.g. "agent_summarizer"
    Each agent can have multiple “skills”; this tells the orchestrator who’s responsible e.g. agent_id="agent_data_loader"

    action = The specific skill or function to run on that agent e.g. "summarize", "analyze_sentiment". 
    Defines what skill the agent should use to process this node e.g. action="summarize_text"
    Corresponds to a function registered via Agent.register_skill()
    The orchestrator will look up: agent._skills["summarize_text"] and execute it with inputs and context.
    
    inputs = IDs of other nodes this node depends on — their results become this node’s inputs. e.g. ["node1", "node2"]
    The orchestrator ensures all input nodes are done before running this node.
    During execution, the results of these input nodes are passed to the payload.
    e.g. inputs=["node_load_article", "node_extract_keywords"]. 
    Means this node will wait for those two tasks to complete and get their results as input.

    metadata = Optional contextual data to pass to the agent — parameters, configuration, or external data 
    e.g. {"temperature": 0.8, "language": "en"}
    Optional configuration, custom parameters, or environment info. Useful for passing extra hints, user instructions, or runtime settings.
    e.g. metadata={
    "user_intent": "summarize for education",
    "output_format": "bulleted"
    }

    The orchestrator automatically passes the output of each completed node into its dependent nodes (the ones that list it in their inputs).
    '''
    def __init__(self, node_id: str, name: str, agent_id: str, action: Optional[str] = None, inputs: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.node_id = node_id
        self.name = name
        self.agent_id = agent_id
        self.action = action or "default"
        self.inputs = inputs or []
        self.metadata = metadata or {}
        self.status = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "name": self.name, "agent_id": self.agent_id, "action": self.action, "inputs": self.inputs, "metadata": self.metadata, "status": self.status, "attempts": self.attempts, "result": ensure_json_serializable(self.result)}

# ---------- Graph Orchestrator (async, cycle detection, persistence)

class GraphOrchestrator:
    '''
    
    
    '''
    def __init__(self, auth: AuthManager, sessions: SessionManager, memory: MemoryManager, persistence: Persistence, concurrency: int = 4) -> None:
        self.auth = auth
        self.sessions = sessions
        self.memory = memory
        self._p = persistence
        self._agents: Dict[str, Agent] = {}
        self._nodes: Dict[str, TaskNode] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(concurrency)
        logger.info("GraphOrchestrator ready (concurrency=%d)", concurrency)
        # load persisted nodes
        # This loads any saved graph from previous runs (so the orchestrator can resume after restart).
        self._load_persisted_nodes()

    def _load_persisted_nodes(self) -> None:
        '''
        This reconstructs the task graph from the database.
        - It creates TaskNode objects from each record.
        - Restores each node’s: status (pending, running, done, error) , attempts , result
        - Rebuilds parent-child relationships (_parents, _children).
        This ensures your system can pause and resume workflows.
        '''
        rows = self._p.load_all_nodes()
        for r in rows:
            node = TaskNode(node_id=r["node_id"], name=r["name"], agent_id=r["agent_id"], action=r["action"], inputs=json.loads(r["inputs"] or "[]"), metadata=json.loads(r["metadata"] or "{}"))
            node.status = r["status"] or "pending"
            node.attempts = r["attempts"] or 0
            node.result = json.loads(r["result"]) if r["result"] else None
            self._nodes[node.node_id] = node
            for p in node.inputs:
                self._children[p].append(node.node_id)
                self._parents[node.node_id].append(p)

    # Agents
    def register_agent(self, agent: Agent) -> None:
        self._agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    # Graph operations with cycle detection
    def add_node(self, node: TaskNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError("node_already_exists")
        # temporary insert for cycle check
        self._nodes[node.node_id] = node
        for p in node.inputs:
            self._children[p].append(node.node_id)
            self._parents[node.node_id].append(p)
        if self._detect_cycle():
            # rollback
            del self._nodes[node.node_id]
            for p in node.inputs:
                self._children[p].remove(node.node_id)
                self._parents[node.node_id].remove(p)
            raise ValueError("cycle_detected")
        # persist
        self._p.save_node(node.to_dict())

    def add_edge(self, parent_id: str, child_id: str) -> None:
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise ValueError("nodes_must_exist")
        self._children[parent_id].append(child_id)
        self._parents[child_id].append(parent_id)
        if self._detect_cycle():
            # rollback
            self._children[parent_id].remove(child_id)
            self._parents[child_id].remove(parent_id)
            raise ValueError("cycle_detected")
        # persist nodes
        self._p.save_node(self._nodes[parent_id].to_dict())
        self._p.save_node(self._nodes[child_id].to_dict())

    def _detect_cycle(self) -> bool:
        '''
        It walks through the graph to ensure it remains a Directed Acyclic Graph (DAG).
        If any node can reach itself again via its children → cycle detected.
        '''
        # classic DFS cycle detection
        visited = set()
        recstack = set()
        def dfs(nid: str) -> bool:
            visited.add(nid)
            recstack.add(nid)
            for child in self._children.get(nid, []):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in recstack:
                    return True
            recstack.remove(nid)
            return False
        for nid in list(self._nodes.keys()):
            if nid not in visited:
                if dfs(nid):
                    return True
        return False

    def _ready_nodes(self) -> List[TaskNode]:
        '''
        Finds all pending nodes whose parent nodes are all done.
        These are ready for execution:
        '''
        ready = []
        for node in self._nodes.values():
            if node.status != "pending":
                continue
            parents = self._parents.get(node.node_id, [])
            if all(self._nodes[p].status == "done" for p in parents):
                ready.append(node)
        return ready

    def _build_payload(self, node: TaskNode) -> Dict[str, Any]:
        '''
        The orchestrator automatically passes the output of each completed node into its dependent nodes 
        (the ones that list it in their inputs).
        That means before a node runs, its payload will contain:
        - The outputs of all its input nodes (as a dictionary)
        - The node’s metadata and action

        When a node runs, its inputs are the results of its parent nodes.
        So an agent receives everything it needs to execute the task, including upstream outputs.
        e.g. payload = {
                "action": node.action,
                "inputs": {parent_id: parent.result, ...},
                "metadata": node.metadata
                }
        '''
        payload_inputs = {}
        for p in node.inputs:
            payload_inputs[p] = self._nodes[p].result
        return {"action": node.action, "inputs": payload_inputs, "metadata": node.metadata}

    async def _execute_node(self, node: TaskNode, session: Dict[str, Any]) -> None:
        '''
        This runs a single task node, asynchronously and safely.
        '''
        async with self._sem: # Concurrency Control: Limits how many nodes run at once.
            # Mark node as running 
            node.status = "running"
            node.attempts += 1
            # Find its assigned agent
            agent = self.get_agent(node.agent_id)
            if not agent:
                node.status = "error"
                node.result = {"status": "error", "error": "missing_agent"}
                self._p.save_node(node.to_dict())
                return
            # Build message for the agent
            msg = Message(frm=session["session_id"], 
                          to=node.agent_id, 
                          task_id=node.node_id, 
                          payload=self._build_payload(node), 
                          metadata={"node_name": node.name, "action": node.action}
                          )
            # Call the agent’s skill
            resp = await agent.handle_message(msg, session, self.memory)
            try:
                validate_agent_response(resp)
            except MessageValidationError as e:
                node.status = "error"
                node.result = {"status": "error", "error": f"invalid_agent_response:{str(e)}"}
                self._p.save_node(node.to_dict())
                return
            # Validate and handle result
            # If response is valid and status == "ok" → mark done
            # Save any returned memories in both short-term and long-term memory.
            # If status == "error" → mark error.
            node.result = resp
            if resp.get("status") == "ok":
                node.status = "done"
                if "memories" in resp:
                    for m in resp["memories"]:
                        self.memory.add_short(session["session_id"], m)
                        self.memory.add_long(session["user_id"], m)
            else:
                node.status = "error"
            # persist node state
            self._p.save_node(node.to_dict())
            # Handle dynamic graph expansion
            # If the agent returns new next_tasks, the orchestrator adds them dynamically:
            # This enables self-extending graphs — e.g., an agent can decide to add more work dynamically based on results.
            for nt in resp.get("next_tasks", []):
                try:
                    nid = nt.get("node_id") or generate_id("dyn")
                    new_node = TaskNode(node_id=nid, 
                                        name=nt.get("name", nid), 
                                        agent_id=nt.get("agent_id"), 
                                        action=nt.get("action"), 
                                        inputs=nt.get("inputs", []),
                                          metadata=nt.get("metadata", {})
                                          )
                    self.add_node(new_node)
                except Exception:
                    logger.exception("Failed to add dynamic node")

    # Executing the Entire Graph
    async def execute_graph(self, session_id: str, max_iterations: int = 1000) -> Dict[str, Any]:
        # This is the main loop that coordinates all node executions.
        # Get the session:
        session = self.sessions.get_session(session_id)
        iterations = 0
        tasks: List[asyncio.Task] = []
        while True:
            if iterations >= max_iterations:
                raise RuntimeError("max_iterations_reached")
            iterations += 1
            # Find all ready nodes
            ready = self._ready_nodes()
            if not ready:
                break
            # For each ready node, schedule _execute_node() asynchronously
            for node in ready:
                t = asyncio.create_task(self._execute_node(node, session))
                tasks.append(t)
            # wait for all scheduled to complete before re-evaluating readiness (simple barrier)
            if tasks:
                # Wait for all current tasks to finish:
                await asyncio.gather(*tasks)
                tasks = []
            # loop continues — newly added nodes will be picked up
            # Newly added nodes (from next_tasks) will appear in the next iteration

        # Stop when no pending nodes remain or max_iterations exceeded
        
        # Return full execution report:
        report = {nid: node.to_dict() for nid, node in self._nodes.items()}
        return {"status": "ok", "report": report}

# ---------- Example skills (async-capable)

async def skill_fetch(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("metadata", {}).get("query") or payload.get("query") or "default"
    await asyncio.sleep(0.1)  # simulate IO
    data = {"fetched_at": iso_now(), "query": query, "items": [f"item_{i}" for i in range(3)]}
    return {"status": "ok", "result": data, "memories": [{"type": "fetch", "detail": data}], "next_tasks": [{"name": "process_results", "agent_id": ctx["session"].get("metadata", {}).get("processor_agent", "agent_processor"), "action": "process", "inputs": [payload.get("metadata", {}).get("origin_node") or "unknown_parent"]}]}

async def skill_process(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.05)
    inputs = payload.get("inputs", {})
    aggregated = {}
    for pid, r in inputs.items():
        if isinstance(r, dict) and "result" in r:
            aggregated[pid] = r["result"]
        else:
            aggregated[pid] = r
    stats = {pid: (len(aggregated[pid].get("items", [])) if isinstance(aggregated[pid], dict) else 1) for pid in aggregated}
    out = {"processed_at": iso_now(), "stats": stats, "summary": f"Processed {sum(stats.values())} items"}
    return {"status": "ok", "result": out, "memories": [{"type": "process_summary", "summary": out["summary"]}]}

def skill_default(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "result": {"note": "default executed", "payload": payload}}

# ---------- Minimal HTTP Server (no external frameworks)

class SimpleHandler(BaseHTTPRequestHandler):
    '''
    This is the HTTP interface that exposes your AI orchestration system (with agents, sessions, and memory) to the outside world
    SimpleHandler is a subclass of Python’s BaseHTTPRequestHandler, which is the core class in the http.server module
    It’s responsible for handling incoming HTTP requests — like POST /create_user or GET /session_status — 
    and routing them to the right part of the backend system.

    '''
    orchestrator: Optional[GraphOrchestrator] = None
    auth: Optional[AuthManager] = None
    session_mgr: Optional[SessionManager] = None
    memory: Optional[MemoryManager] = None

    def _send(self, code: int, data: Any) -> None:
        '''
        A helper function to send JSON responses.
        It writes a JSON-encoded object (data) as the HTTP response. Every route uses this to standardize output.
        Example output:
            {
            "status": "ok",
            "user_id": "user_007"
            }
        '''
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_json(self) -> Any:
        '''
        Reads the incoming HTTP body and parses it as JSON.
        Reads raw bytes from the request (rfile). Converts them to string. Parses JSON safely (returns {} on error).
        Example request body:
        {
            "username": "alice",
            "password": "secure123"
        }
        '''
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_POST(self) -> None:
        '''
        This is where all the API endpoints live. It extracts:
        - the path (e.g. /login, /execute_graph)
        - the body (parsed JSON data)
        Then it matches on each possible endpoint.
        '''
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/create_user":
                # Input: { "username": "alice", "password": "secure123" }
                # Output: { "status": "ok", "user_id": "user_001" }
                username = body["username"]
                password = body["password"]
                user_id = self.auth.create_user(username, password)
                return self._send(200, {"status": "ok", "user_id": user_id})
            if path == "/login": 
                # Authenticates user and returns tokens.
                '''
                Output
                {
                    "status": "ok",
                    "token": "eyJhbGciOi...",
                    "refresh_token": "eyJhbGciOi..."
                }
                '''
                username = body["username"]
                password = body["password"]
                token, refresh = self.auth.login(username, password)
                return self._send(200, {"status": "ok", "token": token, "refresh_token": refresh})
            if path == "/refresh":
                # Refreshes an expired access token using a refresh token.
                refresh = body.get("refresh_token")
                nt, nr = self.auth.refresh_token(refresh)
                return self._send(200, {"status": "ok", "token": nt, "refresh_token": nr})
            if path == "/revoke":
                # Revokes (invalidates) a token — logs out user.
                token = body.get("token")
                self.auth.revoke_token(token)
                return self._send(200, {"status": "ok"})
            if path == "/create_session":
                # This creates a new working session for the user (e.g., a chat, a project, or a run of the graph).
                token = body.get("token")
                uid = self.auth.validate_token(token)
                if not uid:
                    return self._send(401, {"status": "error", "error": "invalid_token"})
                metadata = body.get("metadata", {})
                sid = self.session_mgr.create_session(uid, metadata)
                return self._send(200, {"status": "ok", "session_id": sid})
            if path == "/execute_graph":
                # This endpoint triggers the GraphOrchestrator to run all tasks for a given session.
                session_id = body.get("session_id")
                # run orchestrator execute_graph in event loop thread
                loop = asyncio.get_event_loop()
                res = loop.run_until_complete(self.orchestrator.execute_graph(session_id))
                return self._send(200, res)
            if path == "/recall":
                # This endpoint lets a user query memory.
                token = body.get("token")
                uid = self.auth.validate_token(token)
                if not uid:
                    return self._send(401, {"status": "error", "error": "invalid_token"})
                keywords = body.get("keywords", [])
                session_id = body.get("session_id")
                res = self.memory.recall(uid, keywords, session_id)
                return self._send(200, {"status": "ok", "results": res})
        except Exception as e:
            logger.exception("HTTP handler error")
            return self._send(500, {"status": "error", "error": str(e)})
        return self._send(404, {"status": "error", "error": "not_found"})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        try:
            if path == "/session_status":
                session_id = qs.get("session_id", [None])[0]
                if not session_id:
                    return self._send(400, {"status": "error", "error": "missing_session_id"})
                try:
                    s = self.session_mgr.get_session(session_id)
                    return self._send(200, {"status": "ok", "session": s})
                except SessionExpired as e:
                    return self._send(404, {"status": "error", "error": str(e)})
        except Exception:
            logger.exception("HTTP GET error")
            return self._send(500, {"status": "error"})
        return self._send(404, {"status": "error", "error": "not_found"})

# ---------- Unit Tests

class TestAdvancedOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        # use a temp DB
        self.db_path = "test_orch.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.p = Persistence(self.db_path)
        self.auth = AuthManager(self.p, token_ttl_seconds=2)
        self.sess = SessionManager(self.p, session_ttl_seconds=2)
        self.mem = MemoryManager(self.p)
        self.orch = GraphOrchestrator(self.auth, self.sess, self.mem, self.p, concurrency=2)
        # register agents
        af = Agent("agent_fetcher", "Fetcher")
        af.register_skill("fetch", skill_fetch)
        af.register_skill("default", skill_default)
        ap = Agent("agent_processor", "Processor")
        ap.register_skill("process", skill_process)
        ap.register_skill("default", skill_default)
        self.orch.register_agent(af)
        self.orch.register_agent(ap)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_user_auth_and_token(self):
        uid = self.auth.create_user("bob", "pw")
        token, refresh = self.auth.login("bob", "pw")
        self.assertIsNotNone(self.auth.validate_token(token))
        # wait for expiry
        time.sleep(2.1)
        self.assertIsNone(self.auth.validate_token(token))

    def test_session_and_memory(self):
        uid = self.auth.create_user("carol", "pw2")
        t, r = self.auth.login("carol", "pw2")
        sid = self.sess.create_session(uid, metadata={})
        s = self.sess.get_session(sid)
        self.assertEqual(s["user_id"], uid)
        self.mem.add_short(sid, {"k": "v"})
        short = self.mem.list_short(sid)
        self.assertGreaterEqual(len(short), 1)

    def test_graph_execution_and_cycle_detection(self):
        # add nodes: a -> b -> c, and ensure it runs
        n1 = TaskNode("n1", "fetch", "agent_fetcher", action="fetch", inputs=[], metadata={"query":"x","origin_node":"n1"})
        n2 = TaskNode("n2", "process", "agent_processor", action="process", inputs=["n1"], metadata={})
        self.orch.add_node(n1)
        self.orch.add_node(n2)
        sid = self.sess.create_session(self.auth.create_user("dave","pw3"), {})
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(self.orch.execute_graph(sid))
        self.assertEqual(res["status"], "ok")
        # cycle detection
        n3 = TaskNode("n3","cycle","agent_processor",action="default",inputs=["n2"])
        self.orch.add_node(n3)
        # now add edge from n3 -> n1 which would create cycle
        with self.assertRaises(ValueError):
            self.orch.add_edge("n3", "n1")
 
# ---------- Make live API calls to models such as OpenAI’s, Gemini, or execute commands on a terminal
'''
- real calls to the OpenAI or Gemini APIs
- tool use (web search, bash commands)
- multi-agent collaboration (Planner → Researcher → Executor → Reviewer)
- and integration with your existing GraphOrchestrator.

System Architecture (local runnable sketch)
User → SimpleHandler → GraphOrchestrator
            │
            ├── PlannerAgent  (decides what to do)
            ├── ResearchAgent (calls OpenAI / Gemini APIs)
            ├── ExecAgent     (runs shell tools)
            └── ReviewerAgent (verifies + summarizes)
'''

async def main():
    memory = MemoryManager()
    auth = AuthManager()
    sessions = SessionManager(auth._p)

    orch = GraphOrchestrator(memory, sessions)
    orch.register_agent(planner)
    orch.register_agent(researcher)
    orch.register_agent(executor)
    orch.register_agent(reviewer)

    # Simulate login + session
    uid = "user_001"
    sid = sessions.create_session(uid, {"topic": "AI developments"})
    
    # Construct graph
    nodes = [
        TaskNode("plan1", "Planning", "planner", "plan_research", {"topic": "AI developments"}),
        TaskNode("research1", "Researching", "researcher", "query_llm", {"topic": "AI developments", "query": "Summarize recent AI news"}),
        TaskNode("exec1", "SystemCheck", "executor", "run_command", {"command": "uname -a"}),
        TaskNode("review1", "Review", "reviewer", "summarize_findings", {})
    ]

    # Define edges (Planner → Researcher + Executor → Reviewer)
    edges = [
        ("plan1", "research1"),
        ("plan1", "exec1"),
        ("research1", "review1"),
        ("exec1", "review1")
    ]

    for n in nodes:
        orch.add_node(n)
    for a, b in edges:
        orch.connect(a, b)

    result = await orch.execute_graph(sid)
    print(json.dumps(result, indent=2))

asyncio.run(main())


# ---------- Demo & server run

def run_demo_and_server():
    p = Persistence()
    auth = AuthManager(p)
    sess = SessionManager(p)
    mem = MemoryManager(p)
    orch = GraphOrchestrator(auth, sess, mem, p, concurrency=4)

    # Register agents
    a_fetch = Agent("agent_fetcher", "Fetcher")
    a_fetch.register_skill("fetch", skill_fetch)
    a_fetch.register_skill("default", skill_default)
    a_proc = Agent("agent_processor", "Processor")
    a_proc.register_skill("process", skill_process)
    a_proc.register_skill("default", skill_default)
    orch.register_agent(a_fetch)
    orch.register_agent(a_proc)

    # Setup HTTP handler references
    SimpleHandler.orchestrator = orch
    SimpleHandler.auth = auth
    SimpleHandler.session_mgr = sess
    SimpleHandler.memory = mem

    # Start HTTP server on separate thread
    server = HTTPServer(('localhost', 8080), SimpleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("HTTP server running at http://localhost:8080")

    # create demo user and session
    user_id = auth.create_user("alice", "s3cr3t")
    token, refresh = auth.login("alice", "s3cr3t")
    session_id = sess.create_session(user_id, metadata={"processor_agent": "agent_processor"})

    # create nodes
    n1 = TaskNode("node_fetch_1", "fetch_data", "agent_fetcher", action="fetch", inputs=[], metadata={"query":"latest reports","origin_node":"node_fetch_1"})
    n2 = TaskNode("node_process_1", "process_data", "agent_processor", action="process", inputs=["node_fetch_1"], metadata={})
    orch.add_node(n1)
    orch.add_node(n2)

    # run graph
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(orch.execute_graph(session_id))
    print(json.dumps(res, indent=2))

    # keep server alive for manual testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
        logger.info("Server stopped")

if __name__ == "__main__":
    # run unit tests first
    unittest.main(exit=False)
    # then demo server
    run_demo_and_server()



'''

2. Example: A “Planner” agent that spawns subtasks
3. Example: Conditional dynamic node creation
TODO:
- validate results of node results
- planner agent wihout hard coding next task nodes
Would you like me to show a complete runnable example of this planner + dynamic tasks system (with fake async agents) so you can see the logs and flow in action?


[Planner] Planning for question: What are the effects of climate change on polar bears?


🧩 What is the Message Class?

Would you like me to now show a complete lifecycle of one Message



Going back to GraphOrchestrator, can you illustrate a complete example graph execution with 3 agents (Planner, Worker, Summarizer), where the orchestrator automatically sequences their tasks and stores intermediate results in memory?

Don't understand it:
    async def handle_message(self, message, session, memory):
        skill = message.payload["action"]
        func = self._skills[skill]
        ctx = {"agent": self.name, "session": session, "memory": memory}
        return await func(message.payload, ctx)
GOOD EXAMPLE FOR END-TO-END


🧠 Modified _execute_node() to handle new nodes
🧭 PlannerAgent — Creates tasks dynamically



1. The relevant orchestrator method


Questions
what is the node.action and node.metadata in 
def _build_payload(self, node: TaskNode) -> Dict[str, Any]:
    payload_inputs = {}
    for p in node.inputs:
        payload_inputs[p] = self._nodes[p].result
    return {"action": node.action, "inputs": payload_inputs, "metadata": node.metadata}


how can I call Gemini multiple times using async?




uv versus pip. Why is uv better?
#




MultiServerMCPClient
vs client = await MCPClient.create_stdio(command="python", args=["mcp_calc_server.py"])


Can  you explain what do you mean by the comment
# call in thread since SDK may be sync
at the following code?
            def call_sync():
                return genai.chat.create(model="gemini-pro", messages=[{"role":"user","content": prompt}])
            result = await asyncio.to_thread(call_sync)



# Executor agent: runs shell commands (careful in production)
executor_agent = Agent("executor", "ExecutorAgent")
async def run_shell(payload, ctx):
    cmd = payload.get("metadata", {}).get("cmd") or payload.get("cmd") or payload.get("command", "echo 'no-cmd'")
    # Run command in thread to avoid blocking loop
    def run():
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return {"stdout": p.stdout.strip(), "stderr": p.stderr.strip(), "returncode": p.returncode}
    res = await asyncio.to_thread(run)
    return {"status":"ok", "output": res, "memories":[{"type":"exec","content":res}]}
executor_agent.register_skill("run_shell", run_shell)



p1 = subprocess.Popen(["cat", "file.txt"], stdout=subprocess.PIPE)
p2 = subprocess.Popen(["grep", "keyword"], stdin=p1.stdout, stdout=subprocess.PIPE)
output = p2.communicate()[0]


            if asyncio.iscoroutine(maybe): resp = await maybe
            else: resp = maybe



    # Tool discovery / wrapping
    async def discover_and_wrap_tools(self):
        tools = await self.mcp.list_tools()
        for t in tools:
            name = t.get("name")
            # register a skill on a special tool agent
            async def make_tool_skill(args, ctx, toolname=name):
                # args may be in metadata or payload
                payload_args = args.get("args") or args.get("metadata") or args
                try:
                    out = await self.mcp.call_tool(toolname, payload_args)
                    return {"status":"ok", "tool": toolname, "output": out}
                except Exception as e:
                    return {"status":"error", "error": str(e)}
            # register skill on tool_agent (create if needed)
            ta = self.agents.get("tool_agent")
            if not ta:
                ta = Agent("tool_agent","ToolAgent")
                self.register_agent(ta)
            ta.register_skill(f"tool_{name}", make_tool_skill)
            print("[ORCH] wrapped tool", name, "as skill tool_"+name)

            

# ---------- MCP Client (tiny HTTP client)
class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def list_tools(self) -> List[Dict[str,Any]]:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{self.base_url}/.well-known/tools", timeout=10.0)
            r.raise_for_status()
            return r.json().get("tools", [])

    async def call_tool(self, tool_name: str, args: Dict[str,Any]) -> Dict[str,Any]:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base_url}/call/{tool_name}", json={"args": args}, timeout=20.0)
            if r.status_code >= 400:
                raise RuntimeError(f"tool_call_error: {r.status_code} {r.text}")
            return r.json().get("output", {})
     

# ---------- Pydantic models for agent output validation
class AgentOutputModel(BaseModel):
    status: str
    # free-form other fields allowed
    class Config:
        extra = "allow"

        

        

self._short.setdefault(sid,[]).append(item)


### DOCKERFILES

orchestrator:
  build: ./orchestrator
  container_name: orchestrator
  environment:
    - MCP_URL=http://mcp_server:8000
  depends_on:
    - mcp_server


    
version: "3.8"
services:
  mcp_tool:
    build: ./mcp_tool
    container_name: mcp_tool
    ports:
      - "9000:9000"

  orchestrator:
    build: ./orchestrator
    container_name: orchestrator
    ports:
      - "8000:8000"
    environment:
      - MCP_TOOL_URL=http://mcp_tool:9000
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    depends_on:
      - mcp_tool

### how Docker networking works under the hood

server-sent events (SSE) to simulate MCP streamable HTTP behavior.


proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)


✅ What Are Server-Sent Events (SSE)?
# SSE streaming endpoint for long-running tools
@app.get("/stream/{tool_name}")
async def stream_tool(tool_name: str, request: Request):
    """
    Return a server-sent events stream. We simulate streaming output line-by-line.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="tool_not_found")

    async def event_generator():
        # For demo: stream three messages then finish
        for i in range(3):
            if await request.is_disconnected():
                break
            data = {"part": i+1, "message": f"stream chunk {i+1} from {tool_name}"}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.5)
        # end
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")





TODO: More realistic example, e.g. create an agent that can book a ticket


- Make it specific for OpenAI LLMs, tool calling for those, what is their output and how to schedule next tasks
- How does OpenAI gpt4o decides when to call a skill and when not to?
- 
- Adopt a microservices architecture; Spawn contrainers for each task.
- Show me how to deploy the agent in AWS & GCP and create a running demo to test it


ASYNC/AWAIT tutorial
https://chatgpt.com/c/69186e19-a214-8326-ae55-68531ab61c18
asyncio.iscoroutine



PYDANTIC
AgentOutputModel.parse_obj(resp)


MCP is an open protocol that standardises how external tools, data-sources and services can be exposed to LLM-agents. 
An MCP server exposes tools (functions, APIs, services) via a standard interface; agents act as MCP clients and can discover/list/call those tools.
- build my own MCP server that exposes 1 or more tools
- stdio subprocess, streamable HTTP, WebSocket
- How does the agent decide which tool to call?
- what does registering a skill to an Agent do?
- 





PROMPT TO CREATE AGENTIC FRAMEWORK FROMS SCRATCH





Please do the following:
1) Explain what is the usage of stream_tool and what does the code do. Please give examples of how it works in practice when it is called. At the end re-write it to make it more appropriate for real deployments
2) Make worker.py appropriate for real deployments (passing task via HTTP or mount a task queue instead of using env var TASK_PAYLOAD). What does the worker.py do in practise? Please give practical examples to understand it. 
3) 



# high-level helper: run from a user prompt
    async def run_from_prompt(self, user_id: str, session_id: str, prompt: str):
        # discover tools
        await self.discover_tools_and_wrap()
        # 1) create a planner node (ask openai to plan)
        plan_node = TaskNode(node_id="plan_1", name="PlanStep", agent_id="openai_agent",
          action="ask_gpt4o", 
          metadata={"prompt": f"Given the user request: {prompt}\nReturn a short plan. 
          If you want the orchestrator to call an external tool, include lines like: 
          TOOL: fake_search query=<term> or TOOL: run_shell cmd=<cmd>."})
        self.add_node(plan_node)
        # 2) add an aggregator node that will summarise results at the end (created up front or dynamically)
        agg_node = TaskNode(node_id="agg_1", name="Aggregate", agent_id="openai_agent", action="ask_gpt4o", inputs=["plan_1"], metadata={"prompt":"Summarise outputs: {inputs}"})
        self.add_node(agg_node)
        session = {"session_id": session_id, "user_id": user_id}
        report = await self.execute_graph(session)
        return report








✅ Produce the full microservice architecture with
Orchestrator + Worker + MCP Tool Server + GPT-4o Agent + Reflector Agent + Memory Storage + Graph Execution + Docker Compose.

Or:

✅ Write a fully working multi-agent example using streaming MCP tools.

Just tell me “continue” or what direction you want!



!!!!!!!!!!!!!!!
Note: using docker-py requires the orchestrator container to have access to Docker socket or Docker Engine. This is powerful but dangerous — in production use a controlled job runner / container orchestrator like Kubernetes with proper RBAC.

'''



# SSE streaming endpoint for long-running tools
@app.get("/stream/{tool_name}")
async def stream_tool(tool_name: str, request: Request):
    """
    Return a server-sent events stream. We simulate streaming output line-by-line.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail="tool_not_found")

    async def event_generator():
        # For demo: stream three messages then finish
        for i in range(3):
            if await request.is_disconnected():
                break
            data = {"part": i+1, "message": f"stream chunk {i+1} from {tool_name}"}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.5)
        # end
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")




# worker/worker.py
import os, json, time, sys

def main():
    payload_json = os.environ.get("TASK_PAYLOAD", "{}")
    try:
        payload = json.loads(payload_json)
    except Exception:
        payload = {"error": "invalid_payload"}
    # Simulate work:
    kind = payload.get("kind", "noop")
    time.sleep(1)
    if kind == "compute":
        x = payload.get("x", 1)
        y = payload.get("y", 2)
        out = {"result": x + y}
    elif kind == "echo":
        out = {"echo": payload.get("text", "")}
    else:
        out = {"note": f"no-op for {kind}"}
    # write to stdout as JSON - orchestrator captures container logs
    print(json.dumps({"status":"ok","result": out}))
    sys.exit(0)

if __name__ == "__main__":
    main()





