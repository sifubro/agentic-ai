"""
Agent implementations with multi-model support (GPT-4o, Gemini, Claude fallback),
negotiation protocol, and debate/arbitration systems.
"""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, 
    Union, Awaitable, TYPE_CHECKING
)
import logging
import httpx

from core.types import (
    Message, MessageType, TaskNode, TaskStatus, AgentRole,
    ModelProvider, ModelFallbackError, TaskExecutionError,
    generate_id, current_timestamp
)
from memory.memory_manager import MemoryStore, MultiAgentMemoryManager, MemoryType

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CLIENTS
# ============================================================================

class ModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model is available."""
        pass


class OpenAIClient(ModelClient):
    """OpenAI GPT-4o client."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0
                )
                return response.status_code == 200
        except:
            return False


class GeminiClient(ModelClient):
    """Google Gemini client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using Gemini API."""
        if not self.api_key:
            raise ValueError("Gemini API key not configured")
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens
                    }
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return bool(self.api_key)


class ClaudeClient(ModelClient):
    """Anthropic Claude client."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using Claude API."""
        if not self.api_key:
            raise ValueError("Claude API key not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system_prompt or "",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
    
    async def is_available(self) -> bool:
        """Check if Claude API is available."""
        return bool(self.api_key)


class LocalClient(ModelClient):
    """Local model client for testing."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using local model."""
        # Simple mock response for testing
        return json.dumps({
            "response": f"Local model response to: {prompt[:50]}...",
            "status": "success"
        })
    
    async def is_available(self) -> bool:
        """Check if local model is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=5.0
                )
                return response.status_code == 200
        except:
            return True  # Assume available for testing


class ModelFallbackChain:
    """
    Model fallback chain that tries models in order until one succeeds.
    Implements GPT-4o → Gemini → Claude fallback.
    """
    
    def __init__(
        self,
        openai_key: str = None,
        gemini_key: str = None,
        claude_key: str = None
    ):
        self.clients: Dict[ModelProvider, ModelClient] = {
            ModelProvider.GPT4O: OpenAIClient(openai_key),
            ModelProvider.GEMINI: GeminiClient(gemini_key),
            ModelProvider.CLAUDE: ClaudeClient(claude_key),
            ModelProvider.LOCAL: LocalClient()
        }
        
        self.fallback_order = [
            ModelProvider.GPT4O,
            ModelProvider.GEMINI,
            ModelProvider.CLAUDE,
            ModelProvider.LOCAL
        ]
        
        self._failure_counts: Dict[ModelProvider, int] = {
            p: 0 for p in ModelProvider
        }
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        preferred_model: ModelProvider = None
    ) -> Tuple[str, ModelProvider]:
        """
        Generate response using fallback chain.
        
        Returns:
            Tuple of (response, model_used)
        """
        # Build order starting with preferred model
        order = self.fallback_order.copy()
        if preferred_model and preferred_model in order:
            order.remove(preferred_model)
            order.insert(0, preferred_model)
        
        errors = []
        for provider in order:
            client = self.clients[provider]
            
            # Skip if too many recent failures
            if self._failure_counts[provider] > 3:
                continue
            
            try:
                response = await client.generate(
                    prompt, system_prompt, temperature, max_tokens
                )
                self._failure_counts[provider] = 0
                return response, provider
            except Exception as e:
                self._failure_counts[provider] += 1
                errors.append(f"{provider.value}: {str(e)}")
                logger.warning(f"Model {provider.value} failed: {e}")
        
        raise ModelFallbackError(
            f"All models failed. Errors: {'; '.join(errors)}"
        )
    
    def reset_failure_counts(self):
        """Reset failure counts for all models."""
        self._failure_counts = {p: 0 for p in ModelProvider}


# ============================================================================
# BASE AGENT
# ============================================================================

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        agent_id: str = None,
        name: str = "",
        role: AgentRole = AgentRole.WORKER,
        model_chain: ModelFallbackChain = None,
        memory_manager: MultiAgentMemoryManager = None
    ):
        self.id = agent_id or generate_id()
        self.name = name or f"Agent-{self.id[:8]}"
        self.role = role
        self.model_chain = model_chain or ModelFallbackChain()
        self.memory_manager = memory_manager
        
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._running = False
        self._inbox: asyncio.Queue = asyncio.Queue()
    
    async def initialize(self):
        """Initialize the agent."""
        if self.memory_manager:
            await self.memory_manager.get_or_create_store(self.id)
    
    async def process_message(self, message: Message) -> Message:
        """Process an incoming message."""
        handler = self._message_handlers.get(message.type)
        if handler:
            return await handler(message)
        
        # Default handling
        return await self._default_message_handler(message)
    
    async def _default_message_handler(self, message: Message) -> Message:
        """Default message handler."""
        return Message(
            type=MessageType.RESPONSE,
            sender_id=self.id,
            receiver_id=message.sender_id,
            correlation_id=message.id,
            payload={"status": "received", "original_type": message.type.value}
        )
    
    @abstractmethod
    async def execute_task(self, task: TaskNode) -> TaskNode:
        """Execute a task and return the result."""
        pass
    
    async def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 5
    ) -> Message:
        """Create and return a message to send."""
        return Message(
            type=message_type,
            sender_id=self.id,
            receiver_id=receiver_id,
            payload=payload,
            priority=priority
        )
    
    async def remember(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        tags: List[str] = None
    ):
        """Store something in memory."""
        if self.memory_manager:
            store = await self.memory_manager.get_or_create_store(self.id)
            await store.store(content, memory_type, importance, tags)
    
    async def recall(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories."""
        if self.memory_manager:
            store = await self.memory_manager.get_or_create_store(self.id)
            entries = await store.search_semantic(query, memory_types, top_k)
            return [entry.content for entry in entries]
        return []
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Awaitable[Message]]
    ):
        """Register a message handler."""
        self._message_handlers[message_type] = handler


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class WorkerAgent(BaseAgent):
    """Worker agent that executes tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.WORKER, **kwargs)
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        tool: Callable[..., Awaitable[Any]]
    ):
        """Register a tool for task execution."""
        self.tools[name] = tool
    
    async def execute_task(self, task: TaskNode) -> TaskNode:
        """Execute a task using available tools or LLM."""
        task.status = TaskStatus.RUNNING
        task.started_at = current_timestamp()
        
        try:
            # Check if task specifies a tool
            tool_name = task.input_data.get("tool")
            if tool_name and tool_name in self.tools:
                result = await self.tools[tool_name](**task.input_data.get("args", {}))
                task.output_data = {"result": result}
            else:
                # Use LLM for task
                prompt = self._build_task_prompt(task)
                response, model = await self.model_chain.generate(prompt)
                task.output_data = {
                    "result": response,
                    "model_used": model.value
                }
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = current_timestamp()
            
            # Remember successful execution
            await self.remember(
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "success": True,
                    "output": task.output_data
                },
                importance=0.6,
                tags=["task_execution", "success"]
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = current_timestamp()
            
            # Remember failure
            await self.remember(
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "success": False,
                    "error": str(e)
                },
                importance=0.8,
                tags=["task_execution", "failure"]
            )
        
        return task
    
    def _build_task_prompt(self, task: TaskNode) -> str:
        """Build a prompt for LLM task execution."""
        return f"""Execute the following task:

Task Name: {task.name}
Description: {task.description}

Input Data:
{json.dumps(task.input_data, indent=2)}

Provide a JSON response with the task result."""


class PlannerAgent(BaseAgent):
    """Planner agent that creates and manages execution plans."""
    
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.PLANNER, **kwargs)
    
    async def execute_task(self, task: TaskNode) -> TaskNode:
        """Create a plan for the given task."""
        task.status = TaskStatus.RUNNING
        task.started_at = current_timestamp()
        
        try:
            # Recall similar past plans
            similar_plans = await self.recall(
                task.description,
                memory_types=[MemoryType.PROCEDURAL],
                top_k=3
            )
            
            # Build planning prompt
            prompt = self._build_planning_prompt(task, similar_plans)
            response, model = await self.model_chain.generate(
                prompt,
                system_prompt="You are a strategic planner. Create detailed execution plans in JSON format."
            )
            
            # Parse plan from response
            plan = self._parse_plan(response)
            
            task.output_data = {
                "plan": plan,
                "model_used": model.value
            }
            task.status = TaskStatus.COMPLETED
            task.completed_at = current_timestamp()
            
            # Store plan in procedural memory
            await self.remember(
                {
                    "type": "plan",
                    "task_id": task.id,
                    "plan": plan
                },
                memory_type=MemoryType.PROCEDURAL,
                importance=0.8,
                tags=["plan", task.name]
            )
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = current_timestamp()
        
        return task
    
    def _build_planning_prompt(
        self,
        task: TaskNode,
        similar_plans: List[Dict]
    ) -> str:
        """Build a prompt for plan generation."""
        context = ""
        if similar_plans:
            context = "\nSimilar past plans for reference:\n"
            for plan in similar_plans:
                context += f"- {json.dumps(plan)}\n"
        
        return f"""Create an execution plan for the following task:

Task Name: {task.name}
Description: {task.description}

Input Data:
{json.dumps(task.input_data, indent=2)}
{context}

Provide a JSON response with:
{{
    "steps": [
        {{"id": "step_1", "action": "...", "dependencies": [], "estimated_time": "..."}}
    ],
    "resources_needed": [...],
    "potential_risks": [...],
    "success_criteria": "..."
}}"""
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse plan from LLM response."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        # Return a simple plan if parsing fails
        return {
            "steps": [{"id": "step_1", "action": response, "dependencies": []}],
            "raw_response": response
        }


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages other agents."""
    
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.COORDINATOR, **kwargs)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent to be managed."""
        self.managed_agents[agent.id] = agent
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.managed_agents:
            del self.managed_agents[agent_id]
    
    async def execute_task(self, task: TaskNode) -> TaskNode:
        """Coordinate task execution across agents."""
        task.status = TaskStatus.RUNNING
        task.started_at = current_timestamp()
        
        try:
            # Select best agent for task
            agent = await self._select_agent(task)
            
            if not agent:
                raise TaskExecutionError("No suitable agent available")
            
            # Assign and track
            task.assigned_agent = agent.id
            self._task_assignments[task.id] = agent.id
            
            # Delegate execution
            result = await agent.execute_task(task)
            
            # Cleanup
            del self._task_assignments[task.id]
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = current_timestamp()
            return task
    
    async def _select_agent(self, task: TaskNode) -> Optional[BaseAgent]:
        """Select the best agent for a task."""
        # Get agent workloads
        agent_loads = {
            agent_id: sum(1 for t, a in self._task_assignments.items() if a == agent_id)
            for agent_id in self.managed_agents
        }
        
        # Find agents with matching role
        preferred_role = task.metadata.get("preferred_role", AgentRole.WORKER)
        if isinstance(preferred_role, str):
            preferred_role = AgentRole(preferred_role)
        
        candidates = [
            (agent_id, agent)
            for agent_id, agent in self.managed_agents.items()
            if agent.role == preferred_role
        ]
        
        if not candidates:
            # Fall back to any available agent
            candidates = list(self.managed_agents.items())
        
        if not candidates:
            return None
        
        # Select agent with lowest load
        candidates.sort(key=lambda x: agent_loads.get(x[0], 0))
        return candidates[0][1]


# ============================================================================
# NEGOTIATION AND DEBATE
# ============================================================================

@dataclass
class NegotiationState:
    """State of a negotiation between agents."""
    id: str = field(default_factory=generate_id)
    topic: str = ""
    participants: List[str] = field(default_factory=list)
    proposals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    votes: Dict[str, str] = field(default_factory=dict)  # agent_id -> proposal_id
    status: str = "open"
    winner: Optional[str] = None
    rounds: int = 0
    max_rounds: int = 5


class NegotiationProtocol:
    """Multi-agent negotiation protocol."""
    
    def __init__(self):
        self._negotiations: Dict[str, NegotiationState] = {}
    
    async def start_negotiation(
        self,
        topic: str,
        participants: List[BaseAgent],
        initial_proposals: Dict[str, Dict[str, Any]] = None
    ) -> NegotiationState:
        """Start a new negotiation."""
        state = NegotiationState(
            topic=topic,
            participants=[a.id for a in participants],
            proposals=initial_proposals or {}
        )
        self._negotiations[state.id] = state
        return state
    
    async def submit_proposal(
        self,
        negotiation_id: str,
        agent_id: str,
        proposal: Dict[str, Any]
    ) -> bool:
        """Submit a proposal from an agent."""
        state = self._negotiations.get(negotiation_id)
        if not state or state.status != "open":
            return False
        
        if agent_id not in state.participants:
            return False
        
        proposal_id = generate_id()
        state.proposals[proposal_id] = {
            "agent_id": agent_id,
            "content": proposal,
            "timestamp": current_timestamp()
        }
        return True
    
    async def vote(
        self,
        negotiation_id: str,
        agent_id: str,
        proposal_id: str
    ) -> bool:
        """Vote for a proposal."""
        state = self._negotiations.get(negotiation_id)
        if not state or state.status != "open":
            return False
        
        if agent_id not in state.participants:
            return False
        
        if proposal_id not in state.proposals:
            return False
        
        state.votes[agent_id] = proposal_id
        return True
    
    async def run_round(
        self,
        negotiation_id: str,
        agents: Dict[str, BaseAgent]
    ) -> Optional[str]:
        """Run a negotiation round. Returns winner if consensus reached."""
        state = self._negotiations.get(negotiation_id)
        if not state or state.status != "open":
            return None
        
        state.rounds += 1
        
        # Count votes
        vote_counts: Dict[str, int] = {}
        for proposal_id in state.votes.values():
            vote_counts[proposal_id] = vote_counts.get(proposal_id, 0) + 1
        
        # Check for majority
        majority = len(state.participants) // 2 + 1
        for proposal_id, count in vote_counts.items():
            if count >= majority:
                state.status = "resolved"
                state.winner = proposal_id
                return proposal_id
        
        # Check max rounds
        if state.rounds >= state.max_rounds:
            # Select proposal with most votes
            if vote_counts:
                winner = max(vote_counts.items(), key=lambda x: x[1])[0]
                state.status = "resolved"
                state.winner = winner
                return winner
            state.status = "failed"
        
        return None
    
    def get_state(self, negotiation_id: str) -> Optional[NegotiationState]:
        """Get negotiation state."""
        return self._negotiations.get(negotiation_id)


@dataclass
class DebateState:
    """State of a debate between agents."""
    id: str = field(default_factory=generate_id)
    topic: str = ""
    positions: Dict[str, str] = field(default_factory=dict)  # agent_id -> position
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    rebuttals: List[Dict[str, Any]] = field(default_factory=list)
    arbitrator_id: Optional[str] = None
    verdict: Optional[Dict[str, Any]] = None
    status: str = "open"


class DebateArbitration:
    """Multi-agent debate and arbitration system."""
    
    def __init__(self, model_chain: ModelFallbackChain = None):
        self._debates: Dict[str, DebateState] = {}
        self.model_chain = model_chain or ModelFallbackChain()
    
    async def start_debate(
        self,
        topic: str,
        participants: Dict[str, str],  # agent_id -> position
        arbitrator_id: str = None
    ) -> DebateState:
        """Start a new debate."""
        state = DebateState(
            topic=topic,
            positions=participants,
            arbitrator_id=arbitrator_id
        )
        self._debates[state.id] = state
        return state
    
    async def submit_argument(
        self,
        debate_id: str,
        agent_id: str,
        argument: str,
        supporting_evidence: List[str] = None
    ) -> bool:
        """Submit an argument in the debate."""
        state = self._debates.get(debate_id)
        if not state or state.status != "open":
            return False
        
        if agent_id not in state.positions:
            return False
        
        state.arguments.append({
            "agent_id": agent_id,
            "position": state.positions[agent_id],
            "argument": argument,
            "evidence": supporting_evidence or [],
            "timestamp": current_timestamp()
        })
        return True
    
    async def submit_rebuttal(
        self,
        debate_id: str,
        agent_id: str,
        target_argument_index: int,
        rebuttal: str
    ) -> bool:
        """Submit a rebuttal to an argument."""
        state = self._debates.get(debate_id)
        if not state or state.status != "open":
            return False
        
        if agent_id not in state.positions:
            return False
        
        if target_argument_index >= len(state.arguments):
            return False
        
        state.rebuttals.append({
            "agent_id": agent_id,
            "target_index": target_argument_index,
            "rebuttal": rebuttal,
            "timestamp": current_timestamp()
        })
        return True
    
    async def arbitrate(
        self,
        debate_id: str,
        arbitrator: BaseAgent = None
    ) -> Dict[str, Any]:
        """Have the arbitrator make a decision."""
        state = self._debates.get(debate_id)
        if not state:
            return {"error": "Debate not found"}
        
        # Build context for arbitration
        prompt = self._build_arbitration_prompt(state)
        
        # Use arbitrator's model or fallback chain
        if arbitrator and hasattr(arbitrator, 'model_chain'):
            response, _ = await arbitrator.model_chain.generate(
                prompt,
                system_prompt="You are an impartial arbitrator. Analyze the arguments and evidence to reach a fair verdict."
            )
        else:
            response, _ = await self.model_chain.generate(
                prompt,
                system_prompt="You are an impartial arbitrator. Analyze the arguments and evidence to reach a fair verdict."
            )
        
        # Parse verdict
        verdict = self._parse_verdict(response, state)
        state.verdict = verdict
        state.status = "resolved"
        
        return verdict
    
    def _build_arbitration_prompt(self, state: DebateState) -> str:
        """Build prompt for arbitration."""
        prompt = f"""Topic: {state.topic}

Positions and Arguments:
"""
        for arg in state.arguments:
            prompt += f"\n[{arg['position']}] (Agent {arg['agent_id'][:8]}): {arg['argument']}"
            if arg['evidence']:
                prompt += f"\n  Evidence: {', '.join(arg['evidence'])}"
        
        if state.rebuttals:
            prompt += "\n\nRebuttals:"
            for reb in state.rebuttals:
                prompt += f"\n- Agent {reb['agent_id'][:8]} rebuts argument {reb['target_index']}: {reb['rebuttal']}"
        
        prompt += """

Please provide your verdict in JSON format:
{
    "winning_position": "...",
    "reasoning": "...",
    "strength_scores": {"position1": 0.0-1.0, "position2": 0.0-1.0},
    "key_factors": ["..."]
}"""
        
        return prompt
    
    def _parse_verdict(self, response: str, state: DebateState) -> Dict[str, Any]:
        """Parse verdict from arbitrator response."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        # Default verdict
        positions = list(set(state.positions.values()))
        return {
            "winning_position": positions[0] if positions else "undetermined",
            "reasoning": response,
            "key_factors": []
        }
    
    def get_state(self, debate_id: str) -> Optional[DebateState]:
        """Get debate state."""
        return self._debates.get(debate_id)