# Integration Patterns & Case Studies

Real-world patterns for combining AI reliability tools into production systems.

## Contents

- [Common Integration Patterns](#common-integration-patterns)
- [Pattern 1: Agentic RAG System](#pattern-1-agentic-rag-system)
- [Pattern 2: Production Chatbot](#pattern-2-production-chatbot)
- [Pattern 3: Multi-Agent System](#pattern-3-multi-agent-system)
- [Pattern 4: Enterprise Knowledge Assistant](#pattern-4-enterprise-knowledge-assistant)
- [Pattern 5: Secure API Gateway](#pattern-5-secure-api-gateway)
- [Pattern 6: Research & Experimentation](#pattern-6-research--experimentation)
- [Key Success Factors](#key-success-factors)

---

## Common Integration Patterns

Successful production systems combine multiple tools across different layers:

```
┌─────────────────────────────────────────────────┐
│           User Interface Layer                   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Orchestration & Agent Layer              │
│  (LangChain, CrewAI, AutoGen, LangGraph)        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│          Safety & Guardrails Layer               │
│    (NeMo Guardrails, LLM Guard, Guardrails AI)  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│        LLM Layer (Inference & Serving)           │
│       (vLLM, SGLang, OpenAI API, Ollama)        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    Structured Output & Validation Layer          │
│        (Instructor, Outlines, Guidance)         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│      Memory & Knowledge Layer                    │
│    (Vector DBs, RAG Frameworks, Mem0)           │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│     Observability & Evaluation Layer             │
│   (Langfuse, LangSmith, RAGAS, Promptfoo)       │
└─────────────────────────────────────────────────┘
```

---

## Pattern 1: Agentic RAG System

**Use Case:** Intelligent document Q&A system with autonomous retrieval and reasoning

### Architecture

```
User Query
    ↓
[NeMo Guardrails] ← Input validation
    ↓
[LangGraph] ← Stateful workflow orchestration
    ↓
[Mem0 + Qdrant] ← Temporal knowledge graph + vector search
    ↓
[LlamaIndex] ← Query engine, reranking, retrieval
    ↓
[vLLM] ← High-throughput inference
    ↓
[Instructor] ← Structured output validation
    ↓
[NeMo Guardrails] ← Output filtering
    ↓
[RAGAS + Promptfoo] ← Evaluation (faithfulness, security)
    ↓
[Langfuse] ← Tracing, costs, prompt management
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Orchestration** | LangGraph | State management, human-in-the-loop breakpoints, cyclical workflows |
| **Memory** | Mem0 + Qdrant | Temporal knowledge graph for episodic memory + fast vector search |
| **Retrieval** | LlamaIndex | Sophisticated query engines, reranking, 300+ data connectors |
| **Safety** | NeMo Guardrails | Programmable input/output guards with Colang language |
| **Evaluation** | RAGAS + Promptfoo | RAG-specific metrics + security vulnerability scanning |
| **Observability** | Langfuse | Open-source, span-level tracing, prompt versioning |
| **Serving** | vLLM | PagedAttention for 24x throughput improvement |

### Implementation Highlights

```python
# Example: LangGraph workflow with memory and guardrails
from langgraph.graph import StateGraph
from mem0 import Memory
from nemoguardrails import RailsConfig, LLMRails

# Initialize components
memory = Memory()
rails_config = RailsConfig.from_path("./guardrails")
rails = LLMRails(rails_config)

# Define stateful workflow
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_with_rails)
workflow.add_node("validate", validate_output)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")

# Add memory integration
def retrieve_node(state):
    context = memory.search(state["query"], user_id=state["user_id"])
    vector_results = qdrant.search(state["query"])
    return {"context": merge(context, vector_results)}
```

### Production Metrics

- **Latency:** P95 < 2s for complex multi-hop queries
- **Throughput:** 100+ concurrent users
- **Accuracy:** 95%+ faithfulness score (RAGAS)
- **Cost:** $0.02 per query with caching

---

## Pattern 2: Production Chatbot

**Use Case:** Customer service chatbot with structured outputs and safety guardrails

### Architecture

```
User Message
    ↓
[Guardrails AI] ← PII detection, toxicity filter
    ↓
[LangChain] ← Conversation chains, memory management
    ↓
[OpenAI API / Anthropic Claude] ← LLM inference
    ↓
[Instructor] ← Pydantic validation, type safety
    ↓
[Guardrails AI] ← Output validation, quality checks
    ↓
[DeepEval] ← Continuous evaluation in CI/CD
    ↓
[LiteLLM] ← One-line observability, caching, cost tracking
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Framework** | LangChain | Mature conversation chains, extensive integrations |
| **Structured Outputs** | Instructor | 3M+ monthly downloads, Pydantic validation with retries |
| **Safety** | Guardrails AI | 60+ validators for PII, toxicity, quality criteria |
| **Evaluation** | DeepEval | Pytest integration, automated testing in CI/CD |
| **Observability** | LiteLLM | One-line proxy integration, caching, cost management |
| **Serving** | OpenAI API | Managed service, high reliability, Claude 3.5 Sonnet |

### Implementation Highlights

```python
# Example: Chatbot with structured outputs and guardrails
import instructor
from openai import OpenAI
from guardrails import Guard
from pydantic import BaseModel, Field

# Structured output schema
class CustomerResponse(BaseModel):
    intent: str = Field(description="Detected user intent")
    sentiment: str = Field(description="Positive, Negative, or Neutral")
    response: str = Field(description="Customer-facing response")
    escalate: bool = Field(description="Whether to escalate to human")

# Initialize with guardrails
client = instructor.patch(OpenAI())
guard = Guard.from_rail("chatbot_guardrails.rail")

# Generate structured response
def chat(message: str, history: list):
    # Input validation
    guard.validate("input", message)

    # Structured generation
    response = client.chat.completions.create(
        model="gpt-4",
        response_model=CustomerResponse,
        messages=[{"role": "user", "content": message}]
    )

    # Output validation
    guard.validate("output", response.response)

    return response
```

### Production Metrics

- **Latency:** P95 < 800ms
- **Accuracy:** 92% intent classification
- **Safety:** 99.7% PII blocking rate
- **Cost:** $0.005 per conversation turn with caching

---

## Pattern 3: Multi-Agent System

**Use Case:** Collaborative AI agents for complex task decomposition and execution

### Architecture

```
Complex Task
    ↓
[CrewAI / AutoGen] ← Role-based agent orchestration
    ↓
[MemGPT] ← Hierarchical, self-editing memory
    ↓
[LangSmith] ← Agent-specific metrics, debugging
    ↓
[LLM Guard] ← Input/output scanners for each agent
    ↓
[SGLang / Modal] ← High-throughput or serverless serving
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Orchestration** | CrewAI or AutoGen | Role-based teams (CrewAI) or conversational agents (AutoGen) |
| **Memory** | MemGPT | Hierarchical memory management, self-editing capabilities |
| **Evaluation** | LangSmith | Agent-specific metrics, conversation tracking |
| **Safety** | LLM Guard | CPU-optimized scanners for each agent interaction |
| **Serving** | SGLang or Modal | High throughput (SGLang) or serverless deployment (Modal) |

### Implementation Highlights

```python
# Example: CrewAI multi-agent system
from crewai import Agent, Task, Crew
from memgpt import MemGPT
from llm_guard import scan

# Define specialized agents
researcher = Agent(
    role="Research Analyst",
    goal="Gather and analyze information",
    memory=MemGPT(),
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Create compelling content",
    memory=MemGPT(),
    tools=[writing_tool]
)

# Define collaborative tasks
research_task = Task(
    description="Research topic and gather sources",
    agent=researcher,
    guardrails=scan.scan_prompt
)

writing_task = Task(
    description="Write article based on research",
    agent=writer,
    guardrails=scan.scan_output
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process="sequential"
)

result = crew.kickoff()
```

### Production Metrics

- **Task Success Rate:** 87% for complex multi-step tasks
- **Agent Collaboration:** Average 3.5 agent interactions per task
- **Memory Efficiency:** 90% token savings with MemGPT
- **Latency:** Variable (2-30s depending on complexity)

---

## Pattern 4: Enterprise Knowledge Assistant

**Use Case:** Internal company knowledge base with strict security and compliance

### Architecture

```
Employee Query
    ↓
[Authentication & Authorization]
    ↓
[Promptfoo] ← Security testing (jailbreak, injection)
    ↓
[Semantic Kernel] ← Enterprise orchestration (.NET integration)
    ↓
[Milvus] ← Trillion-scale vector DB with multi-tenancy
    ↓
[vLLM] ← Self-hosted inference for data privacy
    ↓
[Outlines] ← Grammar-based structured outputs
    ↓
[NeMo Guardrails + Lakera Guard] ← Multi-layered safety
    ↓
[LangSmith] ← Enterprise observability & compliance logging
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Orchestration** | Semantic Kernel | Enterprise-ready, .NET support, Microsoft backing |
| **Multi-Agent** | AutoGen | Advanced agent collaboration patterns |
| **Evaluation** | Promptfoo + DeepEval | Security scanning + automated testing |
| **Safety** | NeMo Guardrails + Lakera Guard | Multi-layered defense, ultra-low latency |
| **Vector DB** | Milvus or Qdrant | Enterprise scale, multi-tenancy, self-hosted |
| **Observability** | LangSmith | Enterprise SLA, compliance logging |
| **Serving** | vLLM or SGLang | Self-hosted for data privacy |

### Security Layers

1. **Input Validation:** Promptfoo vulnerability scanning
2. **Prompt Injection Defense:** Lakera Guard (<50ms latency)
3. **Content Moderation:** NeMo Guardrails with custom policies
4. **Output Filtering:** PII detection, data classification
5. **Audit Logging:** LangSmith compliance trails

### Production Metrics

- **Security:** 99.9% attack prevention rate
- **Compliance:** Full audit trails for SOC 2
- **Latency:** P95 < 1.5s with multi-layered security
- **Scale:** 10,000+ concurrent employees

---

## Pattern 5: Secure API Gateway

**Use Case:** Public-facing AI API with rate limiting, safety, and observability

### Architecture

```
External API Request
    ↓
[API Gateway + Rate Limiting]
    ↓
[LiteLLM Proxy] ← Unified interface, caching, load balancing
    ↓
[Guardrails AI] ← Request validation
    ↓
[Multiple LLM Providers] ← OpenAI, Anthropic, Cohere (failover)
    ↓
[Instructor] ← Structured output validation
    ↓
[Guardrails AI] ← Response validation
    ↓
[Langfuse] ← Cost tracking, usage analytics
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Gateway** | LiteLLM Proxy | Multi-provider support, caching, load balancing |
| **Safety** | Guardrails AI | Pre-built validators, custom rules |
| **Observability** | Langfuse | Cost tracking, per-user analytics |
| **Structured Outputs** | Instructor | API-friendly, automatic retries |

### Production Metrics

- **Uptime:** 99.95% with multi-provider failover
- **Cache Hit Rate:** 35% (reduces costs significantly)
- **Rate Limit Compliance:** 100% enforcement
- **Cost per Request:** $0.003 average with caching

---

## Pattern 6: Research & Experimentation

**Use Case:** Academic research or rapid prototyping with local models

### Architecture

```
Research Query
    ↓
[DSPy] ← Automated prompt optimization
    ↓
[Inspect / OpenAI Evals] ← Benchmark evaluation
    ↓
[llama.cpp / vLLM] ← Local inference
    ↓
[Outlines] ← Grammar-based constraints
    ↓
[FAISS / Qdrant] ← Vector search for experiments
    ↓
[Weights & Biases] ← Experiment tracking
```

### Component Selection

| Layer | Tool | Rationale |
|-------|------|-----------|
| **Framework** | DSPy or AutoGen | Systematic optimization vs agent experimentation |
| **Evaluation** | Inspect or OpenAI Evals | Academic benchmarks, reproducibility |
| **Serving** | llama.cpp or vLLM | Local deployment, research-friendly |
| **Vector DB** | FAISS or Qdrant | Research standard vs production-ready |
| **Tracking** | Weights & Biases | ML experiment tracking, visualization |

---

## Key Success Factors

### 1. Structured Outputs
Guarantee downstream integration reliability by constraining generation:
- Use Instructor for API-based systems
- Use Outlines for local/open-source models
- Always validate outputs before downstream use

### 2. Systematic Evaluation
Prevent regressions with continuous testing:
- Unit tests with DeepEval or Pytest
- RAG evaluation with RAGAS
- Security scanning with Promptfoo
- Integration with CI/CD pipelines

### 3. Multi-Layered Safety
Protect against diverse attack vectors:
- Input validation (PII, injections)
- Prompt guardrails (content policies)
- Output filtering (toxicity, quality)
- Rate limiting and authentication

### 4. Comprehensive Observability
Enable debugging and optimization:
- Span-level tracing for all LLM calls
- Cost tracking per user/endpoint
- Prompt versioning and A/B testing
- Error monitoring and alerting

### 5. Efficient Serving
Control costs at scale:
- Use vLLM or SGLang for self-hosted inference
- Implement caching at multiple layers
- Load balance across providers
- Monitor and optimize token usage

### 6. Memory & Context Management
Enable stateful interactions:
- Vector databases for semantic search
- Temporal knowledge graphs for episodic memory
- Conversation history management
- Context window optimization

### 7. Human-in-the-Loop
Maintain control for high-stakes decisions:
- Approval workflows in LangGraph
- Confidence thresholds for escalation
- Audit trails for compliance
- Feedback collection for improvement

### 8. Automated Testing in CI/CD
Catch problems before deployment:
- Regression testing on every commit
- Performance benchmarks
- Security vulnerability scanning
- Canary deployments with gradual rollout

### 9. Cost Tracking
Prevent budget overruns:
- Per-user/per-endpoint cost attribution
- Budget alerts and limits
- Caching to reduce redundant calls
- Model selection based on cost/quality trade-offs

### 10. Vendor Neutrality
Avoid lock-in and enable best-of-breed tooling:
- Use LiteLLM for multi-provider abstraction
- OpenTelemetry for observability
- Standard interfaces (OpenAI-compatible APIs)
- Self-hostable alternatives when possible

---

## Maturity Assessment

### Production-Grade (Enterprise Ready)

**Orchestration:** LangChain, LangGraph, CrewAI, Semantic Kernel
**Inference:** vLLM, SGLang, TGI, TensorRT-LLM
**Structured Outputs:** Instructor, Outlines, Guidance
**Evaluation:** Promptfoo, RAGAS, LangSmith, DeepEval
**Safety:** NeMo Guardrails, Guardrails AI, LLM Guard
**Vector DB:** Milvus, Pinecone, Qdrant, Weaviate
**Observability:** Langfuse, LangSmith, Phoenix

### Rapidly Maturing

AutoGen, PydanticAI, Lunary, Langtrace, OpenLLM, Jan.ai, Modal

### Emerging/Experimental

LangFlow, Botpress, LocalAI, various academic implementations

---

**[⬆ Back to Main README](README.md)** | **[View Comparisons](COMPARISONS.md)** | **[Decision Guide](DECISION_GUIDE.md)**
