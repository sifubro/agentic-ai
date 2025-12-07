PROMPT TO CREATE THE AGENTIC ORCHESTRATOR / PLANNER






------------------------

What worker.py should do in real deployments
A) Receive tasks over HTTP OR a message queue (Like RabbitMQ, Redis streams, Kafka, SQS, NATS, etc.)
B) Process one task (using MCP tools, local tools, or containerized workloads)
C) Stream logs/events back to the orchestrator
D) Post the final result to an orchestrator callback endpoint


validation of input and structured outputs


Graph mutates dynamically


Example:
Step 1 — Planner agent creates tasks:
- Worker container to extract text
- Worker container to chunk text
- Worker container to embed
- GPT-4o summarizer
- Reflector agent


Produce the full microservice architecture with
Orchestrator + Worker + MCP Tool Server + GPT-4o Agent + Reflector Agent + Memory Storage + Graph Execution + Docker Compose.
Write a fully working multi-agent example using streaming MCP tools.

MATTIA









