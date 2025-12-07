"""
Minimal HTTP server implementation without external frameworks.
Provides REST API for the agentic orchestration system.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from typing import (
    Any, Callable, Dict, List, Optional, Pattern, Tuple, 
    Awaitable, Union
)
from urllib.parse import parse_qs, urlparse
import logging

from core.types import (
    Message, MessageType, TaskNode, TaskGraph, Session,
    AuthenticationError, AuthorizationError, generate_id, current_timestamp
)
from auth.authentication import AuthenticationService, TokenManager
from orchestrator.orchestrator import TaskOrchestrator
from graph.task_graph import TaskGraphManager
from storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)


@dataclass
class HTTPRequest:
    """HTTP request representation."""
    method: str = "GET"
    path: str = "/"
    query_params: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    json_body: Optional[Dict[str, Any]] = None
    client_address: Tuple[str, int] = ("", 0)
    
    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")
    
    @property
    def authorization(self) -> Optional[str]:
        auth = self.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None


@dataclass
class HTTPResponse:
    """HTTP response representation."""
    status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    
    @classmethod
    def json(cls, data: Any, status: int = 200) -> 'HTTPResponse':
        """Create JSON response."""
        body = json.dumps(data, default=str).encode('utf-8')
        return cls(
            status=status,
            headers={"Content-Type": "application/json"},
            body=body
        )
    
    @classmethod
    def error(cls, message: str, status: int = 400) -> 'HTTPResponse':
        """Create error response."""
        return cls.json({"error": message}, status)
    
    @classmethod
    def not_found(cls, message: str = "Not found") -> 'HTTPResponse':
        """Create 404 response."""
        return cls.error(message, 404)
    
    @classmethod
    def unauthorized(cls, message: str = "Unauthorized") -> 'HTTPResponse':
        """Create 401 response."""
        return cls.error(message, 401)
    
    def to_bytes(self) -> bytes:
        """Convert response to HTTP bytes."""
        status_text = HTTPStatus(self.status).phrase
        lines = [f"HTTP/1.1 {self.status} {status_text}"]
        
        # Add default headers
        self.headers.setdefault("Content-Length", str(len(self.body)))
        self.headers.setdefault("Connection", "close")
        self.headers.setdefault("Server", "AgenticFramework/1.0")
        
        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")
        
        lines.append("")
        header_bytes = "\r\n".join(lines).encode('utf-8') + b"\r\n"
        return header_bytes + self.body


RouteHandler = Callable[[HTTPRequest], Awaitable[HTTPResponse]]


@dataclass
class Route:
    """Route definition."""
    method: str
    pattern: Pattern
    handler: RouteHandler
    requires_auth: bool = False
    required_scopes: List[str] = field(default_factory=list)


class Router:
    """HTTP router with pattern matching."""
    
    def __init__(self):
        self._routes: List[Route] = []
    
    def add_route(
        self,
        method: str,
        path: str,
        handler: RouteHandler,
        requires_auth: bool = False,
        required_scopes: List[str] = None
    ):
        """Add a route."""
        # Convert path pattern to regex
        pattern = self._path_to_pattern(path)
        route = Route(
            method=method.upper(),
            pattern=pattern,
            handler=handler,
            requires_auth=requires_auth,
            required_scopes=required_scopes or []
        )
        self._routes.append(route)
    
    def get(self, path: str, **kwargs):
        """Decorator for GET routes."""
        def decorator(handler: RouteHandler):
            self.add_route("GET", path, handler, **kwargs)
            return handler
        return decorator
    
    def post(self, path: str, **kwargs):
        """Decorator for POST routes."""
        def decorator(handler: RouteHandler):
            self.add_route("POST", path, handler, **kwargs)
            return handler
        return decorator
    
    def put(self, path: str, **kwargs):
        """Decorator for PUT routes."""
        def decorator(handler: RouteHandler):
            self.add_route("PUT", path, handler, **kwargs)
            return handler
        return decorator
    
    def delete(self, path: str, **kwargs):
        """Decorator for DELETE routes."""
        def decorator(handler: RouteHandler):
            self.add_route("DELETE", path, handler, **kwargs)
            return handler
        return decorator
    
    def match(self, method: str, path: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match a request to a route."""
        for route in self._routes:
            if route.method != method.upper():
                continue
            
            match = route.pattern.match(path)
            if match:
                return route, match.groupdict()
        
        return None
    
    def _path_to_pattern(self, path: str) -> Pattern:
        """Convert path pattern to regex."""
        # Replace {param} with named capture group
        pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', path)
        return re.compile(f"^{pattern}$")


class HTTPServer:
    """
    Minimal async HTTP server implementation.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_service: AuthenticationService = None
    ):
        self.host = host
        self.port = port
        self.router = Router()
        self.auth_service = auth_service
        
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        
        # Middleware chain
        self._middleware: List[Callable] = []
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the chain."""
        self._middleware.append(middleware)
    
    async def start(self):
        """Start the HTTP server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )
        
        addr = self._server.sockets[0].getsockname()
        logger.info(f"HTTP server listening on {addr[0]}:{addr[1]}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the HTTP server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle an incoming connection."""
        try:
            # Parse request
            request = await self._parse_request(reader, writer)
            if not request:
                return
            
            # Apply middleware
            response = None
            for middleware in self._middleware:
                response = await middleware(request)
                if response:
                    break
            
            # Route request
            if not response:
                response = await self._route_request(request)
            
            # Send response
            writer.write(response.to_bytes())
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            try:
                error_response = HTTPResponse.error(str(e), 500)
                writer.write(error_response.to_bytes())
                await writer.drain()
            except:
                pass
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _parse_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> Optional[HTTPRequest]:
        """Parse HTTP request from stream."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(),
                timeout=30.0
            )
            if not request_line:
                return None
            
            request_line = request_line.decode('utf-8').strip()
            parts = request_line.split(' ')
            if len(parts) < 2:
                return None
            
            method = parts[0]
            full_path = parts[1]
            
            # Parse path and query string
            parsed = urlparse(full_path)
            path = parsed.path
            query_params = {
                k: v[0] if len(v) == 1 else v
                for k, v in parse_qs(parsed.query).items()
            }
            
            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                line = line.decode('utf-8').strip()
                if not line:
                    break
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Read body
            body = b""
            content_length = int(headers.get('content-length', 0))
            if content_length > 0:
                body = await reader.read(content_length)
            
            # Parse JSON body if applicable
            json_body = None
            if 'application/json' in headers.get('content-type', ''):
                try:
                    json_body = json.loads(body.decode('utf-8'))
                except:
                    pass
            
            # Get client address
            client_address = writer.get_extra_info('peername', ('', 0))
            
            return HTTPRequest(
                method=method,
                path=path,
                query_params=query_params,
                headers=headers,
                body=body,
                json_body=json_body,
                client_address=client_address
            )
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            return None
    
    async def _route_request(self, request: HTTPRequest) -> HTTPResponse:
        """Route request to handler."""
        # Match route
        match = self.router.match(request.method, request.path)
        if not match:
            return HTTPResponse.not_found(f"No route for {request.method} {request.path}")
        
        route, path_params = match
        
        # Check authentication if required
        if route.requires_auth:
            if not self.auth_service:
                return HTTPResponse.error("Auth service not configured", 500)
            
            token = request.authorization
            if not token:
                return HTTPResponse.unauthorized("Missing authorization token")
            
            auth_token = await self.auth_service.token_manager.validate_token(token)
            if not auth_token:
                return HTTPResponse.unauthorized("Invalid or expired token")
            
            # Check scopes
            if route.required_scopes:
                if not any(scope in auth_token.scopes for scope in route.required_scopes):
                    return HTTPResponse.error("Insufficient permissions", 403)
            
            # Add user info to request
            request.headers['x-user-id'] = auth_token.user_id
        
        # Add path params to query params
        request.query_params.update(path_params)
        
        # Call handler
        try:
            return await route.handler(request)
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return HTTPResponse.error(str(e), 500)


class AgenticAPI:
    """
    REST API for the Agentic Orchestration Framework.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        storage: SQLiteStorage = None,
        auth_service: AuthenticationService = None,
        graph_manager: TaskGraphManager = None,
        orchestrator: TaskOrchestrator = None
    ):
        self.storage = storage
        self.auth_service = auth_service or AuthenticationService(storage)
        self.graph_manager = graph_manager or TaskGraphManager()
        self.orchestrator = orchestrator or TaskOrchestrator(self.graph_manager)
        
        self.server = HTTPServer(host, port, self.auth_service)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        router = self.server.router
        
        # Health check
        router.get("/health")(self._health_check)
        router.get("/ready")(self._ready_check)
        
        # Authentication
        router.post("/auth/register")(self._register)
        router.post("/auth/login")(self._login)
        router.post("/auth/refresh")(self._refresh_token)
        router.post("/auth/logout", requires_auth=True)(self._logout)
        router.get("/auth/me", requires_auth=True)(self._get_current_user)
        
        # Sessions
        router.get("/sessions", requires_auth=True)(self._list_sessions)
        router.post("/sessions", requires_auth=True)(self._create_session)
        router.get("/sessions/{session_id}", requires_auth=True)(self._get_session)
        router.delete("/sessions/{session_id}", requires_auth=True)(self._delete_session)
        
        # Messages
        router.post("/sessions/{session_id}/messages", requires_auth=True)(self._post_message)
        router.get("/sessions/{session_id}/messages", requires_auth=True)(self._get_messages)
        
        # Task Graphs
        router.get("/graphs", requires_auth=True)(self._list_graphs)
        router.post("/graphs", requires_auth=True)(self._create_graph)
        router.get("/graphs/{graph_id}", requires_auth=True)(self._get_graph)
        router.delete("/graphs/{graph_id}", requires_auth=True)(self._delete_graph)
        router.post("/graphs/{graph_id}/execute", requires_auth=True)(self._execute_graph)
        router.get("/graphs/{graph_id}/status", requires_auth=True)(self._get_graph_status)
        
        # Nodes
        router.post("/graphs/{graph_id}/nodes", requires_auth=True)(self._add_node)
        router.get("/graphs/{graph_id}/nodes/{node_id}", requires_auth=True)(self._get_node)
        router.put("/graphs/{graph_id}/nodes/{node_id}", requires_auth=True)(self._update_node)
        router.delete("/graphs/{graph_id}/nodes/{node_id}", requires_auth=True)(self._delete_node)
        
        # Orchestrator
        router.get("/orchestrator/metrics")(self._get_metrics)
        router.get("/orchestrator/workers")(self._get_workers)
    
    async def start(self):
        """Start the API server."""
        if self.storage:
            await self.storage.initialize()
        await self.orchestrator.start()
        await self.server.start()
    
    async def stop(self):
        """Stop the API server."""
        await self.server.stop()
        await self.orchestrator.stop()
        if self.storage:
            await self.storage.close()
    
    # ========================================================================
    # HEALTH ENDPOINTS
    # ========================================================================
    
    async def _health_check(self, request: HTTPRequest) -> HTTPResponse:
        """Health check endpoint."""
        return HTTPResponse.json({
            "status": "healthy",
            "timestamp": current_timestamp()
        })
    
    async def _ready_check(self, request: HTTPRequest) -> HTTPResponse:
        """Readiness check endpoint."""
        return HTTPResponse.json({
            "status": "ready",
            "workers": self.orchestrator.get_pool_status()
        })
    
    # ========================================================================
    # AUTHENTICATION ENDPOINTS
    # ========================================================================
    
    async def _register(self, request: HTTPRequest) -> HTTPResponse:
        """Register a new user."""
        data = request.json_body or {}
        
        username = data.get("username")
        password = data.get("password")
        email = data.get("email")
        
        if not all([username, password, email]):
            return HTTPResponse.error("Missing required fields: username, password, email")
        
        try:
            user = await self.auth_service.register_user(
                username=username,
                password=password,
                email=email,
                roles=data.get("roles", ["user"])
            )
            return HTTPResponse.json({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at
            }, 201)
        except AuthenticationError as e:
            return HTTPResponse.error(str(e), 400)
    
    async def _login(self, request: HTTPRequest) -> HTTPResponse:
        """Login and get tokens."""
        data = request.json_body or {}
        
        username = data.get("username")
        password = data.get("password")
        
        if not all([username, password]):
            return HTTPResponse.error("Missing username or password")
        
        try:
            user, token = await self.auth_service.authenticate(username, password)
            return HTTPResponse.json({
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email
                },
                "access_token": token.token,
                "refresh_token": token.refresh_token,
                "expires_at": token.expires_at
            })
        except AuthenticationError as e:
            return HTTPResponse.unauthorized(str(e))
    
    async def _refresh_token(self, request: HTTPRequest) -> HTTPResponse:
        """Refresh access token."""
        data = request.json_body or {}
        refresh_token = data.get("refresh_token")
        
        if not refresh_token:
            return HTTPResponse.error("Missing refresh_token")
        
        new_token = await self.auth_service.refresh_authentication(refresh_token)
        if not new_token:
            return HTTPResponse.unauthorized("Invalid or expired refresh token")
        
        return HTTPResponse.json({
            "access_token": new_token.token,
            "refresh_token": new_token.refresh_token,
            "expires_at": new_token.expires_at
        })
    
    async def _logout(self, request: HTTPRequest) -> HTTPResponse:
        """Logout and revoke token."""
        token = request.authorization
        await self.auth_service.logout(token)
        return HTTPResponse.json({"status": "logged out"})
    
    async def _get_current_user(self, request: HTTPRequest) -> HTTPResponse:
        """Get current user info."""
        user_id = request.headers.get('x-user-id')
        user = await self.auth_service.get_user(user_id)
        if not user:
            return HTTPResponse.not_found("User not found")
        
        return HTTPResponse.json({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "created_at": user.created_at
        })
    
    # ========================================================================
    # SESSION ENDPOINTS
    # ========================================================================
    
    async def _list_sessions(self, request: HTTPRequest) -> HTTPResponse:
        """List user sessions."""
        user_id = request.headers.get('x-user-id')
        sessions = await self.storage.get_user_sessions(user_id) if self.storage else []
        return HTTPResponse.json([s.to_dict() for s in sessions])
    
    async def _create_session(self, request: HTTPRequest) -> HTTPResponse:
        """Create a new session."""
        user_id = request.headers.get('x-user-id')
        data = request.json_body or {}
        
        session = Session(
            user_id=user_id,
            metadata=data.get("metadata", {})
        )
        
        if self.storage:
            await self.storage.save_session(session)
        
        return HTTPResponse.json(session.to_dict(), 201)
    
    async def _get_session(self, request: HTTPRequest) -> HTTPResponse:
        """Get session details."""
        session_id = request.query_params.get('session_id')
        
        if self.storage:
            session = await self.storage.get_session(session_id)
            if session:
                return HTTPResponse.json(session.to_dict())
        
        return HTTPResponse.not_found("Session not found")
    
    async def _delete_session(self, request: HTTPRequest) -> HTTPResponse:
        """Delete a session."""
        session_id = request.query_params.get('session_id')
        
        if self.storage:
            await self.storage.delete_session(session_id)
        
        return HTTPResponse.json({"status": "deleted"})
    
    # ========================================================================
    # MESSAGE ENDPOINTS
    # ========================================================================
    
    async def _post_message(self, request: HTTPRequest) -> HTTPResponse:
        """Post a message to a session."""
        session_id = request.query_params.get('session_id')
        user_id = request.headers.get('x-user-id')
        data = request.json_body or {}
        
        message = Message(
            type=MessageType(data.get("type", "request")),
            sender_id=user_id,
            receiver_id=data.get("receiver_id", "system"),
            payload=data.get("payload", {})
        )
        
        return HTTPResponse.json(message.to_dict(), 201)
    
    async def _get_messages(self, request: HTTPRequest) -> HTTPResponse:
        """Get messages in a session."""
        session_id = request.query_params.get('session_id')
        # This would need session message storage
        return HTTPResponse.json([])
    
    # ========================================================================
    # GRAPH ENDPOINTS
    # ========================================================================
    
    async def _list_graphs(self, request: HTTPRequest) -> HTTPResponse:
        """List task graphs."""
        user_id = request.headers.get('x-user-id')
        
        if self.storage:
            graphs = await self.storage.list_task_graphs(owner_id=user_id)
            return HTTPResponse.json([g.to_dict() for g in graphs])
        
        return HTTPResponse.json([])
    
    async def _create_graph(self, request: HTTPRequest) -> HTTPResponse:
        """Create a new task graph."""
        user_id = request.headers.get('x-user-id')
        data = request.json_body or {}
        
        graph = await self.graph_manager.create_graph(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            owner_id=user_id
        )
        
        if self.storage:
            await self.storage.save_task_graph(graph)
        
        return HTTPResponse.json(graph.to_dict(), 201)
    
    async def _get_graph(self, request: HTTPRequest) -> HTTPResponse:
        """Get task graph details."""
        graph_id = request.query_params.get('graph_id')
        
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph:
            return HTTPResponse.not_found("Graph not found")
        
        return HTTPResponse.json(graph.to_dict())
    
    async def _delete_graph(self, request: HTTPRequest) -> HTTPResponse:
        """Delete a task graph."""
        graph_id = request.query_params.get('graph_id')
        
        await self.graph_manager.delete_graph(graph_id)
        if self.storage:
            await self.storage.delete_task_graph(graph_id)
        
        return HTTPResponse.json({"status": "deleted"})
    
    async def _execute_graph(self, request: HTTPRequest) -> HTTPResponse:
        """Execute a task graph."""
        graph_id = request.query_params.get('graph_id')
        data = request.json_body or {}
        
        try:
            metrics = await self.orchestrator.execute_graph(
                graph_id,
                timeout=data.get("timeout")
            )
            return HTTPResponse.json(metrics.to_dict())
        except Exception as e:
            return HTTPResponse.error(str(e))
    
    async def _get_graph_status(self, request: HTTPRequest) -> HTTPResponse:
        """Get graph execution status."""
        graph_id = request.query_params.get('graph_id')
        
        metrics = self.orchestrator.get_metrics(graph_id)
        graph = await self.graph_manager.get_graph(graph_id)
        
        return HTTPResponse.json({
            "graph_status": graph.status.value if graph else "unknown",
            "metrics": metrics
        })
    
    # ========================================================================
    # NODE ENDPOINTS
    # ========================================================================
    
    async def _add_node(self, request: HTTPRequest) -> HTTPResponse:
        """Add a node to a graph."""
        graph_id = request.query_params.get('graph_id')
        data = request.json_body or {}
        
        try:
            node = await self.graph_manager.add_node(
                graph_id,
                name=data.get("name", "Task"),
                description=data.get("description", ""),
                dependencies=data.get("dependencies", []),
                input_data=data.get("input_data", {}),
                metadata=data.get("metadata", {}),
                priority=data.get("priority", 5)
            )
            return HTTPResponse.json(node.to_dict(), 201)
        except Exception as e:
            return HTTPResponse.error(str(e))
    
    async def _get_node(self, request: HTTPRequest) -> HTTPResponse:
        """Get node details."""
        graph_id = request.query_params.get('graph_id')
        node_id = request.query_params.get('node_id')
        
        graph = await self.graph_manager.get_graph(graph_id)
        if not graph or node_id not in graph.nodes:
            return HTTPResponse.not_found("Node not found")
        
        return HTTPResponse.json(graph.nodes[node_id].to_dict())
    
    async def _update_node(self, request: HTTPRequest) -> HTTPResponse:
        """Update a node."""
        graph_id = request.query_params.get('graph_id')
        node_id = request.query_params.get('node_id')
        data = request.json_body or {}
        
        node = await self.graph_manager.update_node(graph_id, node_id, data)
        if not node:
            return HTTPResponse.not_found("Node not found")
        
        return HTTPResponse.json(node.to_dict())
    
    async def _delete_node(self, request: HTTPRequest) -> HTTPResponse:
        """Delete a node."""
        graph_id = request.query_params.get('graph_id')
        node_id = request.query_params.get('node_id')
        
        await self.graph_manager.remove_node(graph_id, node_id)
        return HTTPResponse.json({"status": "deleted"})
    
    # ========================================================================
    # ORCHESTRATOR ENDPOINTS
    # ========================================================================
    
    async def _get_metrics(self, request: HTTPRequest) -> HTTPResponse:
        """Get orchestrator metrics."""
        return HTTPResponse.json(self.orchestrator.get_metrics())
    
    async def _get_workers(self, request: HTTPRequest) -> HTTPResponse:
        """Get worker pool status."""
        return HTTPResponse.json(self.orchestrator.get_pool_status())