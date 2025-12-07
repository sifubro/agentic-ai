# LangChain & LangGraph: Complete Multi-Agent Tutorial

## Table of Contents
1. [Foundation: LangChain Basics](#foundation)
2. [ReAct Reasoning Pattern](#react)
3. [Memory Systems](#memory)
4. [RAG (Retrieval-Augmented Generation)](#rag)
5. [LangGraph: State Machines for Agents](#langgraph)
6. [Multi-Agent Collaboration](#multi-agent)
7. [MCP Tools Integration](#mcp)
8. [Advanced Patterns](#advanced)

# ---

## 1. Foundation: LangChain Basics {#foundation}

### Installation
```bash
pip install langchain langchain-openai langchain-anthropic langgraph chromadb faiss-cpu
```

### Basic LLM Call
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Simple call
response = llm.invoke([HumanMessage(content="What is 2+2?")])
print(response.content)
```

### Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert."),
    ("user", "{question}")
])

chain = prompt | llm
response = chain.invoke({
    "role": "mathematics",
    "question": "Explain the Pythagorean theorem"
})
```

### Output Parsers
```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# String parser
chain = prompt | llm | StrOutputParser()

# JSON parser with schema
class Answer(BaseModel):
    explanation: str = Field(description="The explanation")
    confidence: float = Field(description="Confidence score 0-1")

json_parser = JsonOutputParser(pydantic_object=Answer)
chain = prompt | llm | json_parser
```

---

## 2. ReAct Reasoning Pattern {#react}

ReAct (Reasoning + Acting) combines chain-of-thought reasoning with action execution.

### Basic ReAct Agent
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain import hub

# Define tools
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    # Simplified - use actual API in production
    return f"Wikipedia results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Search Wikipedia for factual information"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Perform mathematical calculations"
    )
]

# Get ReAct prompt from hub
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Run agent
result = agent_executor.invoke({
    "input": "What is the square root of the year the Eiffel Tower was completed?"
})
print(result["output"])
```

### Custom ReAct Implementation
```python
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class ReActAgent:
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 10
        
    def run(self, task: str) -> str:
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"Task: {task}")
        ]
        
        for i in range(self.max_iterations):
            response = self.llm.invoke(messages)
            messages.append(response)
            
            # Parse response for action
            if "Action:" in response.content:
                action, action_input = self._parse_action(response.content)
                
                if action == "Finish":
                    return action_input
                
                # Execute tool
                observation = self.tools[action].func(action_input)
                messages.append(HumanMessage(
                    content=f"Observation: {observation}"
                ))
            else:
                break
        
        return "Max iterations reached"
    
    def _get_system_prompt(self) -> str:
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""You are a ReAct agent. Answer the user's task by reasoning step by step.

Available tools:
{tool_descriptions}

Format your response as:
Thought: [your reasoning]
Action: [tool name or "Finish"]
Action Input: [tool input or final answer]

Continue this loop until you can provide a final answer."""

    def _parse_action(self, text: str) -> tuple:
        # Simple parsing - improve for production
        lines = text.split("\n")
        action = None
        action_input = None
        
        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[1].strip()
            elif line.startswith("Action Input:"):
                action_input = line.split("Action Input:")[1].strip()
        
        return action, action_input

# Usage
agent = ReActAgent(llm, tools)
result = agent.run("What is 15 * 7?")
```

---

## 3. Memory Systems {#memory}

### Short-Term Memory (Conversation Buffer)
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Have a conversation
conversation.predict(input="Hi, my name is Alice")
conversation.predict(input="What's my name?")

# View memory
print(memory.load_memory_variables({}))
```

### Window Memory (Last K Messages)
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)  # Keep last 3 exchanges

conversation = ConversationChain(
    llm=llm,
    memory=memory
)
```

### Summary Memory (Long-Term)
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Long conversation - memory will summarize automatically
for i in range(10):
    conversation.predict(input=f"Tell me fact {i} about space")

print(memory.load_memory_variables({}))
```

### Vector Store Memory (Semantic Search)
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Initial memory"],
    embedding=embeddings
)

# Create memory
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Save context
memory.save_context(
    {"input": "My favorite color is blue"},
    {"output": "That's nice!"}
)
memory.save_context(
    {"input": "I work as a software engineer"},
    {"output": "Interesting career!"}
)

# Retrieve relevant memories
print(memory.load_memory_variables({"prompt": "What do I do for work?"})["history"])
```

### Custom Memory Implementation
```python
from langchain.schema import BaseMemory
from typing import Any, Dict, List

class CustomMemory(BaseMemory):
    """Custom memory with importance scoring."""
    
    memories: List[Dict[str, Any]] = []
    
    @property
    def memory_variables(self) -> List[str]:
        return ["history"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Sort by importance and return top memories
        sorted_memories = sorted(
            self.memories,
            key=lambda x: x.get("importance", 0),
            reverse=True
        )[:5]
        
        history = "\n".join([
            f"User: {m['input']}\nAI: {m['output']}"
            for m in sorted_memories
        ])
        
        return {"history": history}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Calculate importance (simplified)
        importance = len(inputs["input"]) / 100.0
        
        self.memories.append({
            "input": inputs["input"],
            "output": outputs["output"],
            "importance": importance
        })
    
    def clear(self) -> None:
        self.memories = []
```

---

## 4. RAG (Retrieval-Augmented Generation) {#rag}

### Basic RAG Pipeline
```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("documents.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain.invoke({"query": "What is the main topic?"})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata}")
```

### Advanced RAG with Re-ranking
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Compressor to re-rank
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use in chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

### Multi-Query RAG
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Generate multiple perspectives of the query
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# This will generate 3-5 variations of your query
docs = multi_query_retriever.get_relevant_documents(
    "What are the benefits?"
)
```

### Parent Document Retriever
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store for parent documents
store = InMemoryStore()

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Larger chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)
```

---

## 5. LangGraph: State Machines for Agents {#langgraph}

LangGraph enables complex agent workflows with state management.

### Basic Graph
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str
    data: dict

# Define nodes
def node_1(state: AgentState) -> AgentState:
    return {
        "messages": ["Executed node 1"],
        "current_step": "node_1",
        "data": {"processed": True}
    }

def node_2(state: AgentState) -> AgentState:
    return {
        "messages": ["Executed node 2"],
        "current_step": "node_2"
    }

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)

workflow.set_entry_point("node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", END)

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [],
    "current_step": "",
    "data": {}
})
print(result)
```

### Conditional Routing
```python
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    question: str
    category: str
    answer: str

def categorize(state: RouterState) -> RouterState:
    question = state["question"]
    
    # Simple categorization
    if "math" in question.lower() or any(c.isdigit() for c in question):
        category = "math"
    elif "code" in question.lower() or "program" in question.lower():
        category = "coding"
    else:
        category = "general"
    
    return {"category": category}

def handle_math(state: RouterState) -> RouterState:
    return {"answer": f"Math answer for: {state['question']}"}

def handle_coding(state: RouterState) -> RouterState:
    return {"answer": f"Coding answer for: {state['question']}"}

def handle_general(state: RouterState) -> RouterState:
    return {"answer": f"General answer for: {state['question']}"}

# Router function
def route(state: RouterState) -> str:
    return state["category"]

# Build graph
workflow = StateGraph(RouterState)

workflow.add_node("categorize", categorize)
workflow.add_node("math", handle_math)
workflow.add_node("coding", handle_coding)
workflow.add_node("general", handle_general)

workflow.set_entry_point("categorize")
workflow.add_conditional_edges(
    "categorize",
    route,
    {
        "math": "math",
        "coding": "coding",
        "general": "general"
    }
)
workflow.add_edge("math", END)
workflow.add_edge("coding", END)
workflow.add_edge("general", END)

app = workflow.compile()

# Test
result = app.invoke({"question": "What is 2+2?", "category": "", "answer": ""})
print(result["answer"])
```

### Agent with Tools in LangGraph
```python
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

@tool
def calculate_tool(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

tools = [search_tool, calculate_tool]
tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def call_tools(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = last_message.tool_calls
    outputs = []
    
    for tool_call in tool_calls:
        tool_result = tool_executor.invoke(
            ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"]
            )
        )
        outputs.append({
            "tool_call_id": tool_call["id"],
            "output": tool_result
        })
    
    return {"messages": outputs}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

---

## 6. Multi-Agent Collaboration {#multi-agent}

### Supervisor Pattern
```python
from typing import Literal
from langchain_core.messages import HumanMessage

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str

# Define specialized agents
def researcher_agent(state: SupervisorState) -> SupervisorState:
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="You are a research specialist. Gather information."),
        *messages
    ])
    return {"messages": [response]}

def writer_agent(state: SupervisorState) -> SupervisorState:
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="You are a writer. Create compelling content."),
        *messages
    ])
    return {"messages": [response]}

def critic_agent(state: SupervisorState) -> SupervisorState:
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="You are a critic. Provide constructive feedback."),
        *messages
    ])
    return {"messages": [response]}

# Supervisor decides routing
def supervisor(state: SupervisorState) -> SupervisorState:
    messages = state["messages"]
    
    prompt = f"""You are a supervisor managing: researcher, writer, critic.
    
Given the conversation, who should act next? Or should we FINISH?

Options: researcher, writer, critic, FINISH

Conversation:
{messages[-1].content if messages else 'Start'}

Respond with only the name."""

    response = llm.invoke([HumanMessage(content=prompt)])
    next_agent = response.content.strip().lower()
    
    return {"next": next_agent}

# Build graph
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)

# Conditional routing
def route(state: SupervisorState) -> str:
    return state["next"]

workflow.set_entry_point("supervisor")

for agent in ["researcher", "writer", "critic"]:
    workflow.add_edge(agent, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "writer": "writer",
        "critic": "critic",
        "finish": END
    }
)

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="Write an article about AI")],
    "next": ""
})
```

### Hierarchical Agent Teams
```python
class TeamState(TypedDict):
    messages: Annotated[list, operator.add]
    team_decision: str
    final_output: str

# Research team
def research_team_supervisor(state: TeamState):
    # Manages junior researchers
    return {"messages": ["Research team completed analysis"]}

# Development team
def dev_team_supervisor(state: TeamState):
    # Manages developers
    return {"messages": ["Development team completed implementation"]}

# Executive supervisor
def executive_supervisor(state: TeamState):
    messages = state["messages"]
    
    # Decide which team to engage
    if "research" in messages[-1].content.lower():
        return {"team_decision": "research"}
    elif "develop" in messages[-1].content.lower():
        return {"team_decision": "dev"}
    else:
        return {"team_decision": "finish"}

# Build hierarchical graph
workflow = StateGraph(TeamState)

workflow.add_node("executive", executive_supervisor)
workflow.add_node("research_team", research_team_supervisor)
workflow.add_node("dev_team", dev_team_supervisor)

workflow.set_entry_point("executive")

def route_teams(state: TeamState) -> str:
    return state["team_decision"]

workflow.add_conditional_edges(
    "executive",
    route_teams,
    {
        "research": "research_team",
        "dev": "dev_team",
        "finish": END
    }
)

workflow.add_edge("research_team", "executive")
workflow.add_edge("dev_team", "executive")

app = workflow.compile()
```

### Collaborative Debate Pattern
```python
class DebateState(TypedDict):
    topic: str
    messages: Annotated[list, operator.add]
    rounds: int
    consensus: bool

def agent_pro(state: DebateState) -> DebateState:
    topic = state["topic"]
    messages = state["messages"]
    
    context = "\n".join([m.content for m in messages[-3:]])
    
    response = llm.invoke([
        SystemMessage(content=f"Argue FOR: {topic}"),
        HumanMessage(content=f"Previous arguments:\n{context}\n\nYour turn:")
    ])
    
    return {"messages": [response]}

def agent_con(state: DebateState) -> DebateState:
    topic = state["topic"]
    messages = state["messages"]
    
    context = "\n".join([m.content for m in messages[-3:]])
    
    response = llm.invoke([
        SystemMessage(content=f"Argue AGAINST: {topic}"),
        HumanMessage(content=f"Previous arguments:\n{context}\n\nYour turn:")
    ])
    
    return {"messages": [response]}

def moderator(state: DebateState) -> DebateState:
    messages = state["messages"]
    rounds = state.get("rounds", 0)
    
    # Check for consensus or max rounds
    if rounds >= 3:
        summary = llm.invoke([
            SystemMessage(content="Summarize this debate and suggest a conclusion"),
            HumanMessage(content="\n".join([m.content for m in messages]))
        ])
        return {"consensus": True, "messages": [summary]}
    
    return {"rounds": rounds + 1}

# Build debate graph
workflow = StateGraph(DebateState)

workflow.add_node("pro", agent_pro)
workflow.add_node("con", agent_con)
workflow.add_node("moderator", moderator)

workflow.set_entry_point("pro")
workflow.add_edge("pro", "con")
workflow.add_edge("con", "moderator")

def should_continue(state: DebateState) -> str:
    return "end" if state.get("consensus") else "continue"

workflow.add_conditional_edges(
    "moderator",
    should_continue,
    {"continue": "pro", "end": END}
)

app = workflow.compile()
```

---

## 7. MCP Tools Integration {#mcp}

Model Context Protocol (MCP) allows agents to interact with external systems.

### Custom MCP Tool
```python
from langchain_core.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class FileSystemInput(BaseModel):
    """Input for file system operations."""
    path: str = Field(description="File path")
    operation: str = Field(description="Operation: read, write, list")
    content: Optional[str] = Field(None, description="Content for write operations")

class FileSystemTool(BaseTool):
    name = "file_system"
    description = "Interact with the file system"
    args_schema: Type[BaseModel] = FileSystemInput
    
    def _run(self, path: str, operation: str, content: Optional[str] = None) -> str:
        """Execute file system operation."""
        import os
        
        if operation == "read":
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        
        elif operation == "write":
            try:
                with open(path, 'w') as f:
                    f.write(content or "")
                return f"Successfully wrote to {path}"
            except Exception as e:
                return f"Error writing file: {e}"
        
        elif operation == "list":
            try:
                return "\n".join(os.listdir(path))
            except Exception as e:
                return f"Error listing directory: {e}"
        
        return "Unknown operation"
    
    async def _arun(self, path: str, operation: str, content: Optional[str] = None) -> str:
        """Async version."""
        return self._run(path, operation, content)

# Use in agent
fs_tool = FileSystemTool()
tools = [fs_tool]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### Database MCP Tool
```python
class DatabaseInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: str = Field(description="Database name")

class DatabaseTool(BaseTool):
    name = "database"
    description = "Query databases"
    args_schema: Type[BaseModel] = DatabaseInput
    
    def _run(self, query: str, database: str) -> str:
        """Execute database query."""
        import sqlite3
        
        try:
            conn = sqlite3.connect(f"{database}.db")
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                return str(results)
            else:
                conn.commit()
                return f"Query executed successfully"
        except Exception as e:
            return f"Database error: {e}"
        finally:
            conn.close()
```

### API Integration MCP Tool
```python
import requests

class APIInput(BaseModel):
    endpoint: str = Field(description="API endpoint URL")
    method: str = Field(description="HTTP method: GET, POST, etc")
    data: Optional[dict] = Field(None, description="Request payload")

class APITool(BaseTool):
    name = "api_client"
    description = "Make HTTP API calls"
    args_schema: Type[BaseModel] = APIInput
    
    def _run(self, endpoint: str, method: str, data: Optional[dict] = None) -> str:
        """Make API request."""
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint)
            elif method.upper() == "POST":
                response = requests.post(endpoint, json=data)
            else:
                return f"Unsupported method: {method}"
            
            return response.text
        except Exception as e:
            return f"API error: {e}"
```

---

## 8. Advanced Patterns {#advanced}

### Self-Correcting Agent
```python
class SelfCorrectingState(TypedDict):
    task: str
    attempts: int
    current_output: str
    feedback: str
    final_output: str

def generate(state: SelfCorrectingState) -> SelfCorrectingState:
    task = state["task"]
    feedback = state.get("feedback", "")
    
    prompt = f"""Task: {task}
    
Previous attempt feedback: {feedback}

Generate an improved response:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_output": response.content,
        "attempts": state.get("attempts", 0) + 1
    }

def evaluate(state: SelfCorrectingState) -> SelfCorrectingState:
    output = state["current_output"]
    task = state["task"]
    
    prompt = f"""Task: {task}
Output: {output}

Is this output satisfactory? If not, provide specific feedback.

Format:
SATISFACTORY: yes/no
FEEDBACK: [your feedback]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    is_satisfactory = "yes" in response.content.lower().split("satisfactory:")[1].split("\n")[0]
    feedback = response.content.split("FEEDBACK:")[1].strip()
    
    return {
        "feedback": feedback,
        "final_output": output if is_satisfactory else ""
    }

def should_continue(state: SelfCorrectingState) -> str:
    if state.get("final_output") or state.get("attempts", 0) >= 3:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(SelfCorrectingState)
workflow.add_node("generate", generate)
workflow.add_node("evaluate", evaluate)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {"continue": "generate", "end": END}
)

app = workflow.compile()
```

### Planning Agent with Dynamic Re-planning
```python
class PlanningState(TypedDict):
    objective: str
    plan: List[str]
    completed_steps: List[str]
    current_step: int
    observations: List[str]

def create_plan(state: PlanningState) -> PlanningState:
    objective = state["objective"]
    observations = state.get("observations", [])
    
    obs_text = "\n".join(observations) if observations else "No previous observations"
    
    prompt = f"""Create a step-by-step plan to achieve: {objective}

Previous observations:
{obs_text}

Provide a numbered list of concrete steps."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse plan into list
    steps = [line.strip() for line in response.content.split("\n") 
             if line.strip() and any(char.isdigit() for char in line[:3])]
    
    return {"plan": steps, "current_step": 0}

def execute_step(state: PlanningState) -> PlanningState:
    plan = state["plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan):
        return {"observations": ["All steps completed"]}
    
    step = plan[current_step]
    
    # Execute step (simplified - would use actual tools)
    prompt = f"Execute this step and report what happened: {step}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    observation = response.content
    
    return {
        "completed_steps": [step],
        "observations": [observation],
        "current_step": current_step + 1
    }

def should_replan(state: PlanningState) -> PlanningState:
    observations = state.get("observations", [])
    plan = state["plan"]
    completed = state.get("completed_steps", [])
    
    if not observations:
        return {"observations": []}
    
    last_obs = observations[-1]
    
    prompt = f"""Plan: {plan}
Completed: {completed}
Last observation: {last_obs}

Should we continue with the current plan or create a new one?
Respond with: CONTINUE or REPLAN"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    needs_replan = "REPLAN" in response.content.upper()
    
    return {"observations": [f"Replan decision: {response.content}"]}

def route_planning(state: PlanningState) -> str:
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    observations = state.get("observations", [])
    
    if current_step >= len(plan):
        return "end"
    
    if observations and "REPLAN" in observations[-1]:
        return "replan"
    
    return "execute"

# Build graph
workflow = StateGraph(PlanningState)

workflow.add_node("create_plan", create_plan)
workflow.add_node("execute_step", execute_step)
workflow.add_node("should_replan", should_replan)

workflow.set_entry_point("create_plan")
workflow.add_edge("create_plan", "execute_step")
workflow.add_edge("execute_step", "should_replan")

workflow.add_conditional_edges(
    "should_replan",
    route_planning,
    {
        "execute": "execute_step",
        "replan": "create_plan",
        "end": END
    }
)

app = workflow.compile()

# Usage
result = app.invoke({
    "objective": "Research and write a blog post about quantum computing",
    "plan": [],
    "completed_steps": [],
    "current_step": 0,
    "observations": []
})
```

### Human-in-the-Loop Agent
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

class HumanApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    pending_action: Optional[dict]
    approved: bool

def agent_action(state: HumanApprovalState) -> HumanApprovalState:
    """Agent proposes an action."""
    messages = state["messages"]
    
    response = llm.invoke(messages)
    
    # Check if this needs approval
    needs_approval = "delete" in response.content.lower() or "send" in response.content.lower()
    
    if needs_approval:
        return {
            "pending_action": {"action": response.content, "approved": False},
            "messages": []
        }
    
    return {"messages": [response]}

def human_approval(state: HumanApprovalState) -> HumanApprovalState:
    """Wait for human approval."""
    pending = state.get("pending_action")
    
    if not pending:
        return {"approved": True}
    
    print(f"\n🔔 ACTION REQUIRES APPROVAL:")
    print(f"   {pending['action']}")
    print("\nApprove? (yes/no): ", end="")
    
    # In production, this would be async/callback-based
    approval = input().strip().lower()
    
    return {
        "approved": approval == "yes",
        "messages": [HumanMessage(content=f"Action {'approved' if approval == 'yes' else 'rejected'}")]
    }

def route_approval(state: HumanApprovalState) -> str:
    if state.get("pending_action"):
        return "approval_needed"
    return "continue"

# Build graph
workflow = StateGraph(HumanApprovalState)

workflow.add_node("agent", agent_action)
workflow.add_node("human", human_approval)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    route_approval,
    {
        "approval_needed": "human",
        "continue": END
    }
)

workflow.add_edge("human", "agent")

# Use memory to persist state
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### Parallel Agent Execution
```python
from typing import List
import asyncio

class ParallelState(TypedDict):
    task: str
    subtasks: List[str]
    results: Annotated[List[str], operator.add]

def split_task(state: ParallelState) -> ParallelState:
    """Split main task into subtasks."""
    task = state["task"]
    
    prompt = f"""Break this task into 3-5 independent subtasks: {task}

Format:
1. Subtask one
2. Subtask two
etc."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    subtasks = [line.strip() for line in response.content.split("\n")
                if line.strip() and any(c.isdigit() for c in line[:3])]
    
    return {"subtasks": subtasks}

async def execute_parallel(state: ParallelState) -> ParallelState:
    """Execute all subtasks in parallel."""
    subtasks = state["subtasks"]
    
    async def process_subtask(subtask: str) -> str:
        response = await llm.ainvoke([HumanMessage(content=f"Complete: {subtask}")])
        return response.content
    
    # Execute all subtasks concurrently
    results = await asyncio.gather(*[process_subtask(st) for st in subtasks])
    
    return {"results": results}

def combine_results(state: ParallelState) -> ParallelState:
    """Combine parallel results."""
    results = state["results"]
    task = state["task"]
    
    combined = "\n\n".join([f"Result {i+1}: {r}" for i, r in enumerate(results)])
    
    prompt = f"""Original task: {task}

Subtask results:
{combined}

Synthesize these into a final answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"results": [response.content]}

# Build graph
workflow = StateGraph(ParallelState)

workflow.add_node("split", split_task)
workflow.add_node("parallel", execute_parallel)
workflow.add_node("combine", combine_results)

workflow.set_entry_point("split")
workflow.add_edge("split", "parallel")
workflow.add_edge("parallel", "combine")
workflow.add_edge("combine", END)

app = workflow.compile()
```

### Streaming Agent with Real-Time Updates
```python
from langchain_core.runnables import RunnableConfig

class StreamingState(TypedDict):
    messages: Annotated[list, operator.add]
    current_thought: str

def streaming_agent(state: StreamingState, config: RunnableConfig) -> StreamingState:
    """Agent that streams thoughts in real-time."""
    messages = state["messages"]
    
    # Stream the response
    thoughts = []
    for chunk in llm.stream(messages):
        thought = chunk.content
        if thought:
            thoughts.append(thought)
            # Emit intermediate state
            yield {"current_thought": "".join(thoughts)}
    
    return {"messages": [AIMessage(content="".join(thoughts))]}

# Build graph
workflow = StateGraph(StreamingState)
workflow.add_node("agent", streaming_agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# Stream results
for update in app.stream({"messages": [HumanMessage(content="Explain quantum computing")]}):
    print(update)
```

---

## 9. Production Patterns {#production}

### Error Handling and Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustState(TypedDict):
    messages: Annotated[list, operator.add]
    errors: List[str]
    retry_count: int

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_llm_call(messages: List) -> str:
    """LLM call with automatic retry."""
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        raise

def safe_agent_node(state: RobustState) -> RobustState:
    """Agent node with error handling."""
    messages = state["messages"]
    retry_count = state.get("retry_count", 0)
    
    try:
        response = robust_llm_call(messages)
        return {
            "messages": [AIMessage(content=response)],
            "retry_count": 0
        }
    except Exception as e:
        error_msg = f"Error after retries: {str(e)}"
        return {
            "errors": [error_msg],
            "retry_count": retry_count + 1
        }

def error_handler(state: RobustState) -> RobustState:
    """Handle errors gracefully."""
    errors = state.get("errors", [])
    
    if errors:
        fallback_response = "I encountered an error. Let me try a different approach."
        return {"messages": [AIMessage(content=fallback_response)]}
    
    return {}

# Build graph
workflow = StateGraph(RobustState)
workflow.add_node("agent", safe_agent_node)
workflow.add_node("error_handler", error_handler)

workflow.set_entry_point("agent")

def check_errors(state: RobustState) -> str:
    return "error" if state.get("errors") else "success"

workflow.add_conditional_edges(
    "agent",
    check_errors,
    {"error": "error_handler", "success": END}
)
workflow.add_edge("error_handler", END)

app = workflow.compile()
```

### Logging and Observability
```python
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObservableState(TypedDict):
    messages: Annotated[list, operator.add]
    metrics: dict

def logged_node(node_name: str):
    """Decorator for adding logging to nodes."""
    def decorator(func):
        def wrapper(state):
            start_time = datetime.now()
            
            logger.info(f"[{node_name}] Starting execution")
            logger.debug(f"[{node_name}] Input state: {json.dumps(state, default=str)}")
            
            try:
                result = func(state)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[{node_name}] Completed in {duration:.2f}s")
                
                # Add metrics
                metrics = state.get("metrics", {})
                metrics[node_name] = {
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                result["metrics"] = metrics
                
                return result
            except Exception as e:
                logger.error(f"[{node_name}] Error: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    return decorator

@logged_node("research")
def research_node(state: ObservableState) -> ObservableState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

@logged_node("analysis")
def analysis_node(state: ObservableState) -> ObservableState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

### Rate Limiting and Cost Control
```python
import time
from collections import deque
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
        self.lock = Lock()
    
    def acquire(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and self.requests[0] < now - 60:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.acquire()
            
            self.requests.append(now)

# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=50)

class CostControlledState(TypedDict):
    messages: Annotated[list, operator.add]
    total_tokens: int
    cost_usd: float

def rate_limited_call(state: CostControlledState) -> CostControlledState:
    """LLM call with rate limiting and cost tracking."""
    messages = state["messages"]
    
    # Respect rate limit
    rate_limiter.acquire()
    
    # Make call with callbacks for token counting
    response = llm.invoke(messages)
    
    # Track costs (example prices)
    input_tokens = len(str(messages)) // 4  # Rough estimate
    output_tokens = len(response.content) // 4
    
    cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)
    
    return {
        "messages": [response],
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost
    }
```

### Caching for Efficiency
```python
from functools import lru_cache
import hashlib

class CachedState(TypedDict):
    query: str
    messages: Annotated[list, operator.add]
    cache_hit: bool

# Simple cache
response_cache = {}

def get_cache_key(messages: List) -> str:
    """Generate cache key from messages."""
    content = str([m.content for m in messages])
    return hashlib.md5(content.encode()).hexdigest()

def cached_llm_node(state: CachedState) -> CachedState:
    """LLM node with response caching."""
    messages = state["messages"]
    cache_key = get_cache_key(messages)
    
    # Check cache
    if cache_key in response_cache:
        logger.info("Cache hit!")
        return {
            "messages": [AIMessage(content=response_cache[cache_key])],
            "cache_hit": True
        }
    
    # Call LLM
    response = llm.invoke(messages)
    
    # Store in cache
    response_cache[cache_key] = response.content
    
    return {
        "messages": [response],
        "cache_hit": False
    }
```

---

## 10. Complete Example: Research Assistant {#example}

Let's build a complete multi-agent research assistant that combines everything.

```python
from typing import List, Dict, Any
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize tools
search = DuckDuckGoSearchRun()
embeddings = OpenAIEmbeddings()

class ResearchState(TypedDict):
    topic: str
    research_plan: List[str]
    search_queries: List[str]
    search_results: List[str]
    documents: List[str]
    vectorstore: Any
    analysis: str
    report: str
    current_step: int

# Node 1: Planning
def create_research_plan(state: ResearchState) -> ResearchState:
    """Create a structured research plan."""
    topic = state["topic"]
    
    prompt = f"""Create a research plan for: {topic}

Generate 3-5 specific aspects to research.
Format each as a clear research question."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    plan = [line.strip() for line in response.content.split("\n")
            if line.strip() and "?" in line]
    
    logger.info(f"Created research plan with {len(plan)} questions")
    return {"research_plan": plan, "current_step": 0}

# Node 2: Query Generation
def generate_search_queries(state: ResearchState) -> ResearchState:
    """Generate search queries from research plan."""
    plan = state["research_plan"]
    current = state.get("current_step", 0)
    
    if current >= len(plan):
        return {"search_queries": []}
    
    question = plan[current]
    
    prompt = f"""Generate 2-3 specific search queries to answer: {question}

Format as a simple list."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    queries = [line.strip().lstrip("- ").lstrip("• ")
               for line in response.content.split("\n")
               if line.strip() and len(line.strip()) > 10][:3]
    
    logger.info(f"Generated {len(queries)} search queries")
    return {"search_queries": queries}

# Node 3: Search Execution
def execute_searches(state: ResearchState) -> ResearchState:
    """Execute web searches."""
    queries = state.get("search_queries", [])
    
    results = []
    for query in queries:
        try:
            result = search.run(query)
            results.append(result)
            logger.info(f"Search completed: {query[:50]}...")
        except Exception as e:
            logger.error(f"Search failed: {e}")
    
    return {"search_results": results}

# Node 4: Document Processing
def process_documents(state: ResearchState) -> ResearchState:
    """Process and vectorize search results."""
    results = state.get("search_results", [])
    
    if not results:
        return {"documents": []}
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    docs = []
    for result in results:
        chunks = splitter.split_text(result)
        docs.extend(chunks)
    
    # Create vector store
    if docs:
        vectorstore = FAISS.from_texts(docs, embeddings)
        logger.info(f"Created vector store with {len(docs)} documents")
        return {
            "documents": docs,
            "vectorstore": vectorstore
        }
    
    return {"documents": []}

# Node 5: Analysis
def analyze_findings(state: ResearchState) -> ResearchState:
    """Analyze collected information."""
    plan = state["research_plan"]
    current = state.get("current_step", 0)
    vectorstore = state.get("vectorstore")
    
    if not vectorstore or current >= len(plan):
        return {"analysis": ""}
    
    question = plan[current]
    
    # Retrieve relevant docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(question)
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""Based on this information, answer: {question}

Information:
{context[:2000]}

Provide a comprehensive answer with key insights."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis": response.content,
        "current_step": current + 1
    }

# Node 6: Report Writing
def write_report(state: ResearchState) -> ResearchState:
    """Compile final research report."""
    topic = state["topic"]
    plan = state["research_plan"]
    
    # Gather all analyses
    all_analyses = []  # In practice, you'd accumulate these
    
    prompt = f"""Write a comprehensive research report on: {topic}

Research questions covered:
{chr(10).join(plan)}

Structure the report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions

Make it professional and well-organized."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    logger.info("Research report completed")
    return {"report": response.content}

# Build the complete workflow
def build_research_assistant():
    workflow = StateGraph(ResearchState)
    
    # Add all nodes
    workflow.add_node("plan", create_research_plan)
    workflow.add_node("queries", generate_search_queries)
    workflow.add_node("search", execute_searches)
    workflow.add_node("process", process_documents)
    workflow.add_node("analyze", analyze_findings)
    workflow.add_node("report", write_report)
    
    # Define flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "queries")
    workflow.add_edge("queries", "search")
    workflow.add_edge("search", "process")
    workflow.add_edge("process", "analyze")
    
    # Loop or finish
    def should_continue(state: ResearchState) -> str:
        current = state.get("current_step", 0)
        plan = state.get("research_plan", [])
        
        if current < len(plan):
            return "continue"
        return "finish"
    
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "continue": "queries",
            "finish": "report"
        }
    )
    
    workflow.add_edge("report", END)
    
    return workflow.compile()

# Usage
research_assistant = build_research_assistant()

result = research_assistant.invoke({
    "topic": "Impact of artificial intelligence on healthcare",
    "research_plan": [],
    "search_queries": [],
    "search_results": [],
    "documents": [],
    "vectorstore": None,
    "analysis": "",
    "report": "",
    "current_step": 0
})

print(result["report"])
```

---

## 11. Best Practices and Tips {#tips}

### Design Patterns

1. **Keep State Minimal**: Only store what you need in state
2. **Make Nodes Idempotent**: Nodes should be safe to retry
3. **Use Type Hints**: Leverage TypedDict for clarity
4. **Separate Concerns**: Each node should have one responsibility
5. **Plan for Failure**: Always include error handling

### Performance Optimization

```python
# Use async when possible
async def fast_parallel_node(state):
    results = await asyncio.gather(
        process_task_1(),
        process_task_2(),
        process_task_3()
    )
    return {"results": results}

# Batch API calls
def batched_calls(items, batch_size=10):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        yield process_batch(batch)

# Cache expensive operations
@lru_cache(maxsize=1000)
def expensive_computation(input_hash):
    return complex_operation(input_hash)
```

### Testing Strategies

```python
import pytest
from unittest.mock import Mock, patch

def test_agent_node():
    """Test individual node behavior."""
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(content="test response")
    
    # Test state
    state = {"messages": [HumanMessage(content="test")]}
    
    # Execute
    result = agent_node(state)
    
    # Assert
    assert len(result["messages"]) > 0
    assert mock_llm.invoke.called

def test_workflow_integration():
    """Test complete workflow."""
    app = build_workflow()
    
    result = app.invoke({"messages": [HumanMessage(content="test")]})
    
    assert "messages" in result
    assert len(result["messages"]) > 0
```

### Monitoring and Debugging

```python
# Add debug visualization
from langgraph.graph import Graph

def visualize_graph(workflow):
    """Generate visual representation."""
    return workflow.get_graph().draw_ascii()

# Add state inspection
def inspect_state(state, node_name):
    """Debug helper for state inspection."""
    print(f"\n=== State at {node_name} ===")
    for key, value in state.items():
        print(f"{key}: {str(value)[:100]}...")
    print("=" * 40)

# Track execution path
execution_path = []

def tracked_node(name):
    def decorator(func):
        def wrapper(state):
            execution_path.append(name)
            return func(state)
        return wrapper
    return decorator
```

---

## Conclusion

This tutorial covered:
- LangChain fundamentals
- ReAct reasoning patterns  
- Multiple memory systems
- RAG implementations
- LangGraph state machines
- Multi-agent architectures
- MCP tool integration
- Production-ready patterns
- Complete working examples

### Next Steps

1. Start with basic chains and gradually add complexity
2. Experiment with different agent patterns
3. Build your own custom tools
4. Monitor performance and costs
5. Iterate based on real-world usage

### Resources

- LangChain Docs: https://python.langchain.com
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangSmith (Observability): https://smith.langchain.com
- Community: https://github.com/langchain-ai/langchain