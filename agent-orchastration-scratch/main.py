"""
Agentic Orchestration Framework (from scratch)

Features implemented:
- Graph-based orchestration where each node is a Task executed by an Agent
- Agents communicate via structured JSON Messages and JSON Responses
- Session management with TTL
- Memory management (short-term per-session + long-term per-user) with simple keyword recall
- Authentication manager (user creation, login, token validation) with secure password hashing
- Message parsing + validation (no external libraries, JSON-only interchange)
- Simple in-memory persistence and optional disk write/read for long-term memory
- Dynamic task creation support: agents can return `next_tasks` in JSON responses that the orchestrator will insert into the graph

No external libraries used — only Python standard library.

Example usage is at the bottom under `if __name__ == "__main__"` which demonstrates:
- creating users and authenticating
- creating agents and registering skills
- building a simple two-node graph (fetch -> process)
- running the orchestrator and printing the JSON outputs

This is a foundational, educational implementation — not production hardened.
"""

import json
import time
import secrets
import hashlib
import hmac
import base64
import threading
import logging
import uuid
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("AgenticOrchestrator")

# ---------- Utilities

'''
Walkthrough — file organization & utilities

Top-level imports & logging

Only standard library modules used: json, time, uuid, secrets, hashlib, hmac, threading, etc.

Logging configured for visibility; useful during development.

Utility functions

now_ts() / iso_now() — unified timestamp helpers.

generate_id(prefix) — short unique ids for nodes/sessions/users to keep things traceable.

ensure_json_serializable(obj) — lightweight conversion helper to ensure stored results / memories are JSON-compatible (handles set, objects with __dict__, otherwise str() fallbacks). This avoids runtime serialization errors when storing in memory or returning results.

Why: Small, deterministic helpers reduce duplication and help keep stored data consistent and serializable.
'''

def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_json_serializable(obj: Any) -> Any:
    """Try to convert known non-serializable types to serializable ones.
    This is lightweight — users should return simple JSON-friendly types.
    """
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return {k: ensure_json_serializable(v) for k, v in obj.__dict__.items()}
        return str(obj)


# ---------- Authentication Manager

class AuthError(Exception):
    pass


class AuthManager:
    '''
    Purpose: create users, log in, and manage short-lived tokens.
    **Key points:**
    - Stores users in an in-memory dict: {username: {user_id, pw_hash, salt, created_at}}.
    - Password hashing uses HMAC-SHA256(password, salt) stored as hex. Salt is random per user.
    - Login issues a token (secrets.token_urlsafe(32)) with expiry now + token_ttl.
    - validate_token(token) returns the user_id if valid, else None.
    - revoke_token(token) allows invalidation.

    **Design trade-offs / notes:**
        - Simple, standard-library approach avoids external dependencies.
        - This is not as resistant to offline brute-force or as feature-rich as PBKDF2/Argon2 — PBKDF2/Argon2 and salted iterations would be recommended for production.
        - Tokens are stored in memory (no DB), so restarting the process loses active sessions/tokens.
    '''
    def __init__(self, token_ttl_seconds: int = 3600):
        self._users: Dict[str, Dict[str, Any]] = {}  # username -> {user_id, pw_hash, salt, created_at}
        self._tokens: Dict[str, Dict[str, Any]] = {}  # token -> {user_id, expires_at}
        self._token_ttl = token_ttl_seconds
        logger.info("AuthManager initialized (token ttl %ds)", token_ttl_seconds)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        # Use HMAC-SHA256 for password hashing with salt
        return hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()

    def create_user(self, username: str, password: str) -> str:
        if username in self._users:
            raise AuthError("username_already_exists")
        user_id = generate_id("user")
        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)
        self._users[username] = {
            "user_id": user_id,
            "pw_hash": pw_hash,
            "salt": salt,
            "created_at": now_ts(),
        }
        logger.info("Created user %s (id=%s)", username, user_id)
        return user_id

    def login(self, username: str, password: str) -> str:
        if username not in self._users:
            raise AuthError("invalid_credentials")
        record = self._users[username]
        expected_hash = record["pw_hash"]
        salt = record["salt"]
        if not hmac.compare_digest(expected_hash, self._hash_password(password, salt)):
            raise AuthError("invalid_credentials")
        token = secrets.token_urlsafe(32)
        expires_at = now_ts() + self._token_ttl
        self._tokens[token] = {"user_id": record["user_id"], "expires_at": expires_at}
        logger.info("User %s logged in (token=%s) expires=%s", username, token, datetime.utcfromtimestamp(expires_at).isoformat())
        return token

    def validate_token(self, token: str) -> Optional[str]:
        info = self._tokens.get(token)
        if not info:
            return None
        if now_ts() > info["expires_at"]:
            del self._tokens[token]
            return None
        return info["user_id"]

    def revoke_token(self, token: str):
        if token in self._tokens:
            del self._tokens[token]


# ---------- Session Management

class SessionExpired(Exception):
    pass


class SessionManager:
    '''
    Purpose: manage ephemeral sessions tied to users.
    Key points:
    - Each session is a dict {session_id, user_id, created_at, expires_at, metadata, graph}.
    - create_session(user_id, metadata) creates a session and stores TTL.
    - get_session(session_id) validates expiry and returns session; raises SessionExpired if missing/expired.
    - touch_session extends TTL; end_session deletes it.

    Why: Sessions provide scoping for short-term memory and graph execution. Session metadata is used for passing orchestration hints (e.g., which processor agent to use).
    '''
    def __init__(self, session_ttl_seconds: int = 3600):
        self._sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> {user_id, created_at, expires_at, metadata}
        self._session_ttl = session_ttl_seconds
        logger.info("SessionManager initialized (ttl %ds)", session_ttl_seconds)

    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = generate_id("sess")
        now = now_ts()
        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + self._session_ttl,
            "metadata": metadata or {},
            "graph": None,  # orchestrator will attach graph
        }
        logger.info("Created session %s for user %s", session_id, user_id)
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        s = self._sessions.get(session_id)
        if not s:
            raise SessionExpired("session_not_found")
        if now_ts() > s["expires_at"]:
            del self._sessions[session_id]
            raise SessionExpired("session_expired")
        return s

    def touch_session(self, session_id: str):
        s = self.get_session(session_id)
        s["expires_at"] = now_ts() + self._session_ttl

    def end_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]


# ---------- Memory Management

class MemoryManager:
    """Simple memory manager:
    - short-term memory: per-session circular buffer (max_items)
    - long-term memory: per-user list persisted optionally to disk (simple JSON)
    - recall: basic keyword matching in stored "text" fields

    Memory manager (MemoryManager)
    Purpose: two-tier memory: short-term (session-scoped) and long-term (user-scoped).
    Implementation:
    -Short-term: per-session deque(maxlen=short_term_max) for recent records.
    -Long-term: per-user list appended; optional JSON file persistence if long_term_path provided.
    - add_short(session_id, item) and add_long(user_id, item, persist=False).
    - recall(user_id, keywords, session_id=None, limit=10) — naive keyword matching:
        --serializes each memory entry with json.dumps(...).lower()
        --returns matches containing any keyword (searches short-term first if session_id provided)

    Why and tradeoffs:
    - Keeps memory simple and easy to inspect.
    - Keyword matching is fast and dependency-free but not semantically rich; if you want semantic recall use embeddings and a vector index.
    - Persistence is simple JSON dump — fine for prototypes, not recommended for heavy production use.
    """

    def __init__(self, short_term_max: int = 50, long_term_path: Optional[str] = None):
        self._short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=short_term_max))
        self._long_term: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._long_term_path = long_term_path
        if long_term_path and os.path.exists(long_term_path):
            try:
                with open(long_term_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._long_term.update(data)
                    logger.info("Loaded long-term memory from %s", long_term_path)
            except Exception:
                logger.exception("Failed loading long-term memory file; starting empty")

    def add_short(self, session_id: str, item: Dict[str, Any]):
        item_copy = ensure_json_serializable(item)
        self._short_term[session_id].append({"ts": now_ts(), "item": item_copy})
        logger.debug("Short-term memory added for %s: %s", session_id, item_copy)

    def list_short(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._short_term.get(session_id, []))

    def add_long(self, user_id: str, item: Dict[str, Any], persist: bool = False):
        item_copy = ensure_json_serializable(item)
        self._long_term[user_id].append({"ts": now_ts(), "item": item_copy})
        logger.debug("Long-term memory added for %s: %s", user_id, item_copy)
        if persist and self._long_term_path:
            try:
                with open(self._long_term_path, "w", encoding="utf-8") as f:
                    json.dump(self._long_term, f, indent=2)
                logger.info("Persisted long-term memory to %s", self._long_term_path)
            except Exception:
                logger.exception("Failed to persist long-term memory")

    def recall(self, user_id: str, keywords: List[str], session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Return items that contain any of the keywords in their string representation.
        This is naive keyword matching but effective for small-scale demos.
        """
        results: List[Tuple[float, Dict[str, Any]]] = []
        def match_item(item):
            '''
            This function:
            - Takes a single argument item.
            - Converts that item (whatever data structure it is) into a lowercased string representation.
            - Checks whether any of the words in a list called keywords appear in that string.
            - Returns True if there’s at least one match, otherwise False.
            '''

            #Converts the item (which might be a dict, list, or other JSON-serializable object) into a JSON-formatted string 
            text = json.dumps(item, ensure_ascii=False).lower()
            # This is a generator expression that loops through all kw in the list (or iterable) keywords.
            # For each keyword kw: kw.lower() ensures the keyword is also lowercased.
            # (kw.lower() in text) checks if the keyword appears anywhere in the string representation of item
            # any(...) returns: True if at least one keyword is found in the text.  False if no keywords are found
            return any(kw.lower() in text for kw in keywords)

        # search short-term first (if session_id provided)
        if session_id:
            # Iterates through the deque from most recent to oldest (because we want the latest matches first)
            # So this loop walks through the most recent records stored for that session
            for rec in reversed(self._short_term.get(session_id, [])):
                # Each rec is a dictionary. checks if that content contains any of the keywords (as we explained earlier).
                # If it matches, we process it further.
                if match_item(rec["item"]):
                    # [(1728383823, {"message": "hello"}),(1728383821, {"message": "python"}),]
                    results.append((rec["ts"], rec["item"]))
                    #If we’ve already collected enough matches (limit), we don’t need to keep searching.
                    # This makes the search efficient, stopping early instead of scanning everything.
                    if len(results) >= limit:
                        # We return just the items, not the timestamps. r[1] extracts the second element from each (ts, item) tuple.
                        return [r[1] for r in results]

        # search long-term (user-level)
        # self._long_term is a defaultdict(list) mapping:  user_id (string) ➝ a list of records.
        for rec in reversed(self._long_term.get(user_id, [])):
            if match_item(rec["item"]):
                results.append((rec["ts"], rec["item"]))
                if len(results) >= limit:
                    break
        return [r[1] for r in results[:limit]]


# ---------- Messaging / Parsing

class MessageValidationError(Exception):
    pass


class Message:
    '''
    Message class
    - Standardized message schema: required keys from, to, task_id, payload.
    - Methods: to_dict(), to_json(), and parse(raw) accepts dict or JSON string.
    - Timestamps are recorded (ts).
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
        return {
            "from": self.frm,
            "to": self.to,
            "task_id": self.task_id,
            "payload": ensure_json_serializable(self.payload),
            "metadata": ensure_json_serializable(self.metadata),
            "ts": self.ts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def parse(raw: Any) -> "Message":
        # Accept dict or JSON string
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


def validate_agent_response(resp: Dict[str, Any]):
    '''
    Agent response validation
    - validate_agent_response(resp) ensures response is an object with a status key ("ok" or "error").
    - Optionally allows result, next_tasks, memories.
    - next_tasks must be a list (if present).

    Why: By forcing structured messages and responses we can orchestrate reliably between agents and detect malformed behavior early.
    '''
    # Basic validation for agent responses (must be JSON-serializable and contain `status` key)
    if not isinstance(resp, dict):
        raise MessageValidationError("response_must_be_object")
    if "status" not in resp:
        raise MessageValidationError("response_missing_status")
    if resp["status"] not in {"ok", "error"}:
        raise MessageValidationError("response_status_invalid")
    # result is optional but if present must be serializable
    if "result" in resp:
        ensure_json_serializable(resp["result"])
    # next_tasks optional: list of node descriptors
    if "next_tasks" in resp:
        if not isinstance(resp["next_tasks"], list):
            raise MessageValidationError("next_tasks_must_be_list")


# ---------- Agent model

class Agent:
    '''
    Purpose: encapsulate skills that can be invoked by name.

    Key features:
    - register_skill(name, func) — attach callable skill functions.
    - handle_message(message, session, memory) — decides which skill to call (looks at message.metadata[skill] or payload[action] or falls back to 'default'), builds ctx and executes the skill.
    - ctx contains agent identity, session, message metadata, memory manager reference, and timestamp.

    Skill functions are expected to have signature:
        def skill(payload: dict, ctx: dict) -> dict
    and return validated JSON-like dicts, e.g.:
        { "status":"ok", "result": {...}, "memories":[...], "next_tasks":[...] }
    Why: This pattern gives clear separation: 
    - agents implement domain logic; 
    - orchestrator controls flow. 
    - ctx gives access to session/memory without coupling agents to orchestrator internals.
    '''
    def __init__(self, agent_id: str, name: str, role: str = "assistant"):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self._skills: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}
        logger.info("Agent created: %s (%s)", name, agent_id)

    def register_skill(self, name: str, func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
        """Register a skill function.
        The function signature should be func(payload: dict, ctx: dict) -> dict (JSON-like response)
        """
        self._skills[name] = func
        logger.debug("Agent %s registered skill %s", self.agent_id, name)

    def handle_message(self, message: Message, session: Dict[str, Any], memory: MemoryManager) -> Dict[str, Any]:
        logger.info("Agent %s handling message %s -> %s", self.agent_id, message.frm, message.to)
        # Decide which skill to call: first check metadata.skill then payload.action
        skill = message.metadata.get("skill") or message.payload.get("action")
        if not skill:
            # fallback to default skill 'default'
            skill = "default"
        if skill not in self._skills:
            logger.warning("Agent %s missing skill %s", self.agent_id, skill)
            return {"status": "error", "error": f"missing_skill:{skill}"}
        # Build context
        ctx = {
            "agent": {"id": self.agent_id, "name": self.name, "role": self.role},
            "session": session,
            "message_meta": message.metadata,
            "ts": now_ts(),
            "memory": memory,
        }
        try:
            result = self._skills[skill](message.payload, ctx)
            # Validate response
            if not isinstance(result, dict):
                return {"status": "error", "error": "skill_returned_non_object"}
            validate_agent_response(result)
            return result
        except Exception as e:
            logger.exception("Agent skill %s raised", skill)
            return {"status": "error", "error": f"skill_exception:{str(e)}"}


# ---------- Task Node + Graph

class TaskNode:
    '''
    Represents one unit of work in the graph.

    Fields:
    - node_id, name, agent_id, action (skill name), inputs (parent node ids), metadata.
    - Runtime fields: status (pending, running, done, error), result, attempts.

    .to_dict() returns JSON-friendly view.

    Role in graph: nodes declare dependencies (inputs) and are executed when all parents are done.
    '''
    def __init__(self, node_id: str, name: str, agent_id: str, action: Optional[str] = None, inputs: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.name = name
        self.agent_id = agent_id
        self.action = action or "default"
        self.inputs = inputs or []  # list of node_ids that are prerequisites
        self.metadata = metadata or {}
        self.status = "pending"  # pending, running, done, error
        self.result: Optional[Dict[str, Any]] = None
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "agent_id": self.agent_id,
            "action": self.action,
            "inputs": self.inputs,
            "status": self.status,
            "attempts": self.attempts,
            "result": ensure_json_serializable(self.result),
        }


class GraphOrchestrator:

    
    """Orchestrates execution of TaskNodes connected as a directed acyclic graph (DAG).
    - Nodes are added and edges are implied via node.inputs
    - Agents are registered and executed synchronously
    - Agents return JSON responses; orchestrator stores them and can add new nodes dynamically

    Graph orchestrator (GraphOrchestrator)

    This is the core of coordination. Main responsibilities:

    1. Agent & node registration

    - register_agent(agent), add_node(node) and add_edge(parent, child).
    - Maintains _children and _parents adjacency maps for quick lookup.

    2. Readiness detection

    - _ready_nodes() iterates nodes and returns those pending whose parents are all done.
    - This provides a simple DAG execution semantics.

    3. Execution loop (execute_graph(session_id))

    - Attaches graph snapshot to session.
    - Loop:
        - Find ready nodes.
        - For each ready node:

            - Mark running; increment attempts.
            - Find agent (error if missing).
            - Build node payload with parent results:
            - payload = {"action": node.action, "inputs": { parent_id: parent.result }, "metadata": node.metadata}
            - Wrap in a Message and call agent.handle_message(msg, session, memory).
            - Validate response.
            - On status == "ok" -> mark done and optionally store memories.
            - On status != "ok" -> mark error.
            - Handle next_tasks: agent can return list of node descriptors and the orchestrator creates new TaskNode instances and adds them to graph (dynamic graph expansion).
    - The loop repeats until no ready nodes remain (or max_iterations exceeded).
    - Returns a report mapping node_ids to node dicts.

    Edge cases handled:
    - Missing agent -> node goes error.
    - Malformed agent response -> node goes error.
    - next_tasks insertion happens at runtime — new nodes become eligible in subsequent iterations.
    - There is a max_iterations guard to avoid an infinite loop.

    What is not implemented (explicitly):
    - Cycle detection (if an agent creates a node that depends on itself or forms a cycle the loop could deadlock or repeatedly attempt — max_iterations helps but is not a proper cycle check).
    - Parallel execution — everything runs synchronously in the current implementation.
    - Robust persistence/transactionality across restarts.
    """

    def __init__(self, auth: AuthManager, sessions: SessionManager, memory: MemoryManager):
        self.auth = auth
        self.sessions = sessions
        self.memory = memory
        self._agents: Dict[str, Agent] = {}
        self._nodes: Dict[str, TaskNode] = {}
        # adjacency lists
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)
        logger.info("GraphOrchestrator initialized")

    # Agent management
    def register_agent(self, agent: Agent):
        self._agents[agent.agent_id] = agent
        logger.info("Registered agent %s (%s)", agent.name, agent.agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    # Node/graph management
    def add_node(self, node: TaskNode):
        if node.node_id in self._nodes:
            raise ValueError("node_already_exists")
        self._nodes[node.node_id] = node
        for p in node.inputs:
            self._children[p].append(node.node_id)
            self._parents[node.node_id].append(p)
        logger.info("Added node %s (agent=%s) inputs=%s", node.node_id, node.agent_id, node.inputs)

    def add_edge(self, parent_id: str, child_id: str):
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise ValueError("nodes_must_exist")
        self._children[parent_id].append(child_id)
        self._parents[child_id].append(parent_id)

    def _ready_nodes(self) -> List[TaskNode]:
        # nodes that are pending and whose parents are all done
        ready = []
        for node in self._nodes.values():
            if node.status != "pending":
                continue
            parents = self._parents.get(node.node_id, [])
            if all(self._nodes[p].status == "done" for p in parents):
                ready.append(node)
        return ready

    def _build_payload_for_node(self, node: TaskNode) -> Dict[str, Any]:
        # It takes one argument:
        # node — a TaskNode instance that represents the task that’s about to be executed.
        # This function's job is to build the JSON payload that will be sent to the agent responsible for executing this node.
        # Gather results from parents into payload.inputs[parent_id] = parent.result
        # In a task graph, a node may have one or more dependencies (inputs) — 
        # meaning this node should only run after those parent nodes are done, and it should receive their results.
        payload_inputs = {} #This will store the outputs of all parent nodes.
        # We loop over each parent node ID listed in the inputs attribute of the current TaskNode
        for p in node.inputs: # For each parent p
            # We look up that parent node object in the orchestrator’s self._nodes dictionary
            # We grab its .result (this is the JSON response returned by the agent that executed that parent task).
            # We store it under the parent’s ID in the payload dictionary.
            payload_inputs[p] = self._nodes[p].result

        #the method returns the final structured payload that will be passed to the agent handling the current node
        # "action" — what skill or behavior the agent should execute. (e.g., "process")
        # "inputs" — a mapping from parent node IDs to their outputs/results.
        # "metadata" — the metadata attached to the current node (e.g., parameters, config, context).
        return {"action": node.action, "inputs": payload_inputs, "metadata": node.metadata}

    def execute_graph(self, session_id: str, max_iterations: int = 1000) -> Dict[str, Any]:
        # Attach graph snapshot to session for visibility
        session = self.sessions.get_session(session_id)
        session_graph = {nid: node.to_dict() for nid, node in self._nodes.items()}
        session["graph"] = session_graph

        iterations = 0
        while True:
            if iterations >= max_iterations:
                raise RuntimeError("max_iterations_reached")
            iterations += 1
            ready = self._ready_nodes()
            if not ready:
                break
            for node in ready:
                node.status = "running"
                node.attempts += 1
                agent = self.get_agent(node.agent_id)
                if not agent:
                    node.status = "error"
                    node.result = {"status": "error", "error": "missing_agent"}
                    logger.error("Node %s has missing agent %s", node.node_id, node.agent_id)
                    continue
                # Create message
                msg = Message(frm=session_id, to=node.agent_id, task_id=node.node_id, payload=self._build_payload_for_node(node), metadata={"node_name": node.name, "action": node.action})
                # Call agent
                resp = agent.handle_message(msg, session, self.memory)
                # Validate
                try:
                    validate_agent_response(resp)
                except MessageValidationError as e:
                    node.status = "error"
                    node.result = {"status": "error", "error": f"invalid_agent_response:{str(e)}"}
                    logger.error("Invalid response from agent for node %s: %s", node.node_id, e)
                    continue
                # Store result
                node.result = resp
                if resp.get("status") == "ok":
                    node.status = "done"
                    # Optionally write memories if response instructs so
                    if "memories" in resp:
                        mems = resp["memories"]
                        if isinstance(mems, list):
                            for m in mems:
                                self.memory.add_short(session_id, m)
                                self.memory.add_long(session["user_id"], m)
                else:
                    node.status = "error"
                # Handle dynamic next_tasks
                next_tasks = resp.get("next_tasks") or []
                for nt in next_tasks:
                    try:
                        # nt expected to be mapping like {"name":..., "agent_id":..., "action":..., "inputs": [...]} 
                        nid = nt.get("node_id") or generate_id("dyn")
                        new_node = TaskNode(node_id=nid, name=nt.get("name", nid), agent_id=nt.get("agent_id"), action=nt.get("action"), inputs=nt.get("inputs", []), metadata=nt.get("metadata", {}))
                        self.add_node(new_node)
                    except Exception:
                        logger.exception("Failed adding dynamic next_task %s", nt)
            # continue loop — newly added nodes will be considered in next iteration
        # finished; build report
        report = {nid: node.to_dict() for nid, node in self._nodes.items()}
        logger.info("Graph execution finished in %d iterations", iterations)
        return {"status": "ok", "report": report}


# ---------- Example / Demo Skills

# Skill functions take payload (dict) and ctx (dict) and return a dict response with keys:
# - status: 'ok' | 'error'
# - result: optional payload
# - next_tasks: optional list of nodes to add
# - memories: optional list of memory items to store


def skill_fetch(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Demonstration: pretend to fetch data based on query
    inputs = payload.get("inputs", {})
    query = payload.get("metadata", {}).get("query") or payload.get("query") or "default query"
    # produce faux data
    data = {"fetched_at": iso_now(), "query": query, "items": [f"item_{i}" for i in range(3)]}
    result = {"status": "ok", "result": data}
    # store a short memory and offer a next task 'process'
    result["memories"] = [{"type": "fetch", "detail": data}]
    # next task instructs the orchestrator to create a processing node that depends on this node
    result["next_tasks"] = [
        {"name": "process_results", "agent_id": ctx["session"].get("metadata", {}).get("processor_agent", "agent_processor"), "action": "process", "inputs": [payload.get("metadata", {}).get("origin_node") or "unknown_parent"]}
    ]
    return result


def skill_process(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Process data from inputs
    inputs = payload.get("inputs", {})
    # flatten parent results
    aggregated = {}
    for pid, r in inputs.items():
        if not r:
            continue
        if isinstance(r, dict) and "result" in r:
            aggregated[pid] = r["result"]
        else:
            aggregated[pid] = r
    # naive processing: count items
    stats = {pid: (len(aggregated[pid].get("items", [])) if isinstance(aggregated[pid], dict) else 1) for pid in aggregated}
    out = {"processed_at": iso_now(), "stats": stats, "summary": f"Processed {sum(stats.values())} items"}
    return {"status": "ok", "result": out, "memories": [{"type": "process_summary", "summary": out["summary"]}]}


def skill_default(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    # Generic fallback skill
    return {"status": "ok", "result": {"note": "default skill executed", "payload": payload}}


# ---------- Simple test / demo runner

if __name__ == "__main__":
    # Instantiate managers
    auth = AuthManager(token_ttl_seconds=3600)
    sessions = SessionManager(session_ttl_seconds=1800)
    memory = MemoryManager(short_term_max=100, long_term_path=None)

    # Create a user and login
    user_id = auth.create_user("alice", "s3cr3t")
    token = auth.login("alice", "s3cr3t")
    validated_user = auth.validate_token(token)
    print("Logged in user id:", validated_user)

    # Create a session tied to user
    session_id = sessions.create_session(validated_user, metadata={"processor_agent": "agent_processor"})
    session_obj = sessions.get_session(session_id)

    # Create orchestrator and agents
    orchestrator = GraphOrchestrator(auth=auth, sessions=sessions, memory=memory)

    # Agent that fetches data
    a_fetch = Agent(agent_id="agent_fetcher", name="Fetcher")
    a_fetch.register_skill("fetch", skill_fetch)
    a_fetch.register_skill("default", skill_default)

    # Agent that processes data
    a_proc = Agent(agent_id="agent_processor", name="Processor")
    a_proc.register_skill("process", skill_process)
    a_proc.register_skill("default", skill_default)

    orchestrator.register_agent(a_fetch)
    orchestrator.register_agent(a_proc)

    # Build initial nodes
    n1 = TaskNode(node_id="node_fetch_1", name="fetch_data", agent_id="agent_fetcher", action="fetch", inputs=[], metadata={"query": "latest reports", "origin_node": "node_fetch_1"})
    # For demo, we add a processing node that depends on the fetch
    n2 = TaskNode(node_id="node_process_1", name="process_data", agent_id="agent_processor", action="process", inputs=["node_fetch_1"], metadata={})

    orchestrator.add_node(n1)
    orchestrator.add_node(n2)

    # Execute graph
    result = orchestrator.execute_graph(session_id)

    print("Execution result (JSON):\n")
    print(json.dumps(result, indent=2))

    # Inspect memories
    print("\nShort-term memory for session:\n", json.dumps(memory.list_short(session_id), indent=2))
    print("\nLong-term memory for user:\n", json.dumps(memory._long_term.get(user_id, []), indent=2))

    # Demonstrate recall
    recalled = memory.recall(user_id, keywords=["reports", "summary"], session_id=session_id)
    print("\nRecall results:\n", json.dumps(recalled, indent=2))
