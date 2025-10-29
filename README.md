# Awesome AI Reliability Tools

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of awesome tools, frameworks, and techniques for building reliable AI systems. This list covers prompt engineering, structured outputs, evaluation tools, agent orchestration, guardrails, and more.

## Additional Resources

📊 **[Tool Comparisons](COMPARISONS.md)** - Detailed comparison tables across all categories
🏗️ **[Integration Patterns](INTEGRATION_PATTERNS.md)** - Real-world architecture patterns and case studies
🧭 **[Decision Guide](DECISION_GUIDE.md)** - Framework for selecting the right tools for your needs

## Contents

- [Structured Output & Reliability](#structured-output--reliability)
- [Prompting & Instruction Optimization](#prompting--instruction-optimization)
- [Evaluation, Testing & Debugging](#evaluation-testing--debugging)
- [Orchestration & Agents](#orchestration--agents)
- [Guardrails, Moderation & Safety](#guardrails-moderation--safety)
- [Memory, Knowledge & Retrieval](#memory-knowledge--retrieval)
  - [Vector Databases](#vector-databases)
  - [RAG Frameworks](#rag-frameworks)
- [Monitoring & Observability](#monitoring--observability)
- [Model Serving & Hosting](#model-serving--hosting)
  - [Performance Leaders](#performance-leaders)
  - [Local Deployment](#local-deployment)
- [Academic Research](#academic-research)
- [Resources](#resources)
  - [Key Papers](#key-papers)

**[⬆ back to top](#contents)**

## Structured Output & Reliability

*Tools for ensuring LLMs produce valid, typed outputs with grammar-based constraints or schema validation.*

- [Outlines](https://github.com/outlines-dev/outlines) - Structured text generation with regex and Pydantic model support.
- [Guidance](https://github.com/guidance-ai/guidance) - Language for controlling LLMs with interleaved generation and control flow.
- [Instructor](https://github.com/jxnl/instructor) - Type-safe data extraction from LLMs using Pydantic models with automatic retries.
- [LMQL](https://github.com/eth-sri/lmql) - SQL-like query language for programming LLMs with decoding-time constraints.
- [PydanticAI](https://github.com/pydantic/pydantic-ai) - Agent framework from the Pydantic team focused on type safety and structured outputs.
- [TypeChat](https://github.com/microsoft/TypeChat) - Library that translates natural language requests into validated JSON.
- [Mirascope](https://github.com/Mirascope/mirascope) - AI engineering framework for building reliable, scalable, and observable AI systems.
- [BAML](https://github.com/BoundaryML/baml) - Generates structured JSON with fixed structural tokens filled by the library.
- [SGLang](https://github.com/sgl-project/sglang) - PyTorch-like LLM pipeline optimization framework.

**[⬆ back to top](#contents)**

## Prompting & Instruction Optimization

*Frameworks and tools for systematic prompt engineering, management, and optimization.*

- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for programming LLMs with automated prompt optimization.
- [PromptLayer](https://github.com/MagnivOrg/promptlayer-python) - Platform for prompt management, collaboration, and evaluation with versioning and A/B testing.
- [LangChain Hub](https://github.com/hwchase17/langchain) - Robust system for creating and managing prompt templates within LangChain.
- [TextPrompt](https://github.com/jina-ai/textprompt) - Python library for solving NLP tasks using LLMs with programmatic prompt templates.
- [PromptBase](https://promptbase.com) - Marketplace for buying and selling high-quality prompts for DALL·E, Midjourney, and GPT.

**[⬆ back to top](#contents)**

## Evaluation, Testing & Debugging

*Tools for testing, evaluating, and monitoring LLM applications with comprehensive benchmarks and metrics.*

- [Promptfoo](https://github.com/promptfoo/promptfoo) - Testing and evaluation tool for LLM prompts with CLI and CI/CD integration.
- [TruLens](https://github.com/truera/trulens) - Library for evaluating and tracing LLM applications, especially RAG systems.
- [Langfuse](https://github.com/langfuse/langfuse) - Open-source LLM engineering platform for observability, evals, and prompt management.
- [LiteLLM](https://github.com/BerriAI/litellm) - Open-source observability platform and LLM proxy for logging and cost management.
- [DeepEval](https://github.com/confident-ai/deepeval) - Open-source LLM evaluation framework similar to Pytest for unit testing.
- [RAGAS](https://github.com/explodinggradients/ragas) - Framework for evaluating RAG pipelines with metrics like faithfulness and context precision.
- [LangSmith](https://www.langchain.com/langsmith) - Platform for debugging, testing, and monitoring LLM applications with LangChain integration.
- [Giskard](https://github.com/Giskard-AI/giskard) - Python library for automatic detection of performance, bias, and security issues.
- [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) - Agent evaluation framework with sandboxed execution for 50+ benchmarks.
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) - Agent platform with integrated evaluation pipeline.
- [OpenAI Evals](https://github.com/openai/evals) - Open-source framework and registry of benchmarks for evaluating model performance.

**[⬆ back to top](#contents)**

## Orchestration & Agents

*Frameworks for building and orchestrating LLM-powered applications and multi-agent systems.*

- [LangChain](https://github.com/langchain-ai/langchain) - Comprehensive framework for developing LLM applications with extensive integrations.
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data-centric framework for connecting LLMs to external data through RAG pipelines.
- [CrewAI](https://github.com/joaomdmoura/crewai) - Framework for orchestrating role-playing autonomous AI agents in collaborative crews.
- [AutoGen](https://github.com/microsoft/autogen) - Framework from Microsoft for multi-agent applications with conversational collaboration.
- [Flowise](https://github.com/FlowiseAI/Flowise) - Drag-and-drop UI to build customized LLM flows.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Enterprise-ready orchestration framework from Microsoft for AI agents.
- [LangGraph](https://github.com/langchain-ai/langgraph) - Library for building stateful, multi-actor applications with LLMs on top of LangChain.
- [Haystack](https://github.com/deepset-ai/haystack) - Open-source framework for building customizable, production-ready LLM applications.
- [PydanticAI](https://github.com/pydantic/pydantic-ai) - Type-safe agent framework from the Pydantic team.
- [Dust](https://github.com/dust-tt/dust) - Enterprise AI agent platform for connecting internal knowledge, tools, and workflows.

**[⬆ back to top](#contents)**

## Guardrails, Moderation & Safety

*Tools for implementing safety checks, content moderation, and protection against adversarial attacks.*

- [Guardrails AI](https://github.com/guardrails-ai/guardrails) - Framework with 60+ pre-built validators for PII, toxicity, and quality criteria.
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Programmable guardrails using Colang language with five rail types.
- [LLM Guard](https://github.com/protectai/llm-guard) - Toolkit with 15+ scanners optimized for CPU inference.
- [Rebuff](https://github.com/protectai/rebuff) - Multi-layered defense framework against prompt injection with self-hardening.
- [LLM Firewall](https://github.com/aiassured/llm-firewall) - Self-hardening firewall for LLMs protecting against adversarial attacks.

**[⬆ back to top](#contents)**

## Memory, Knowledge & Retrieval

*Vector databases, RAG frameworks, and memory systems for LLM applications.*

### Vector Databases

- [Milvus](https://github.com/milvus-io/milvus) - High-performance vector database with 11+ index types for trillion-vector scale.
- [Pinecone](https://www.pinecone.io) - Managed vector database with serverless architecture and low latency.
- [Qdrant](https://github.com/qdrant/qdrant) - Hybrid cloud vector database written in Rust with high performance.
- [Weaviate](https://github.com/weaviate/weaviate) - Vector database with knowledge graph capabilities and GraphQL API.
- [Chroma](https://github.com/chroma-core/chroma) - Lightweight open-source embedding database for conversational AI.
- [FAISS](https://github.com/facebookresearch/faiss) - Library from Meta AI for efficient similarity search and clustering of dense vectors.
- [LanceDB](https://github.com/lancedb/lancedb) - Serverless vector database built on Lance columnar format for multimodal data.

### RAG Frameworks

- [LlamaIndex](https://github.com/run-llama/llama_index) - Data-centric RAG framework with 300+ integrations and sophisticated query engines.
- [Mem0](https://github.com/mem0ai/mem0) - Memory layer with temporal knowledge graph for AI agents.
- [MemGPT](https://github.com/cpacker/MemGPT) - Framework enabling LLM agents to manage their own long-term memory.

**[⬆ back to top](#contents)**

## Monitoring & Observability

*Platforms for tracing, logging, and monitoring LLM applications with OpenTelemetry support.*

- [Langfuse](https://github.com/langfuse/langfuse) - Comprehensive open-source platform with span-level tracing and prompt management.
- [Phoenix](https://github.com/Arize-ai/phoenix) - Vendor-agnostic platform built on OpenTelemetry with evaluation library.
- [LangSmith](https://www.langchain.com/langsmith) - Platform with deep LangChain integration for annotation and evaluation.
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM gateway providing observability, caching, and cost tracking.
- [AgentOps](https://github.com/AgentOps-AI/agentops) - Open-source platform for logging, tracing, and automated agent optimization.
- [Weights & Biases Weave](https://github.com/wandb/weave) - LLM-specific toolkit for experiment tracking and model versioning.
- [Lunary](https://github.com/lunary-ai/lunary) - Open-source platform focused on chatbot applications with conversation tracking.

**[⬆ back to top](#contents)**

## Model Serving & Hosting

*Inference optimization and hosting solutions for running LLMs efficiently.*

### Performance Leaders

- [SGLang](https://github.com/sgl-project/sglang) - Fast inference engine with RadixAttention for massive throughput gains.
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput inference library with PagedAttention for up to 24x speedup.
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - Rust, Python, and gRPC server powering HuggingFace production services.

### Local Deployment

- [Ollama](https://github.com/ollama/ollama) - CLI-based local inference with GGUF support and one-line model management.
- [LM Studio](https://lmstudio.ai/) - GUI application for local model deployment with Apple MLX support.
- [Jan](https://github.com/janhq/jan) - Open-source, offline-first alternative to ChatGPT.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Zero-dependency C/C++ foundation for running LLMs on edge devices.

**[⬆ back to top](#contents)**

## Academic Research

*Foundational papers and research on LLM reasoning, tool use, and alignment.*

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Technique for step-by-step reasoning to improve multi-step task performance.
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) - Paradigm combining reasoning with external tool use.
- [Toolformer](https://arxiv.org/abs/2302.04761) - Self-supervised learning approach for LLMs to use tools.
- [Self-Consistency](https://arxiv.org/abs/2203.11171) - Improved CoT by sampling multiple reasoning paths with majority voting.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - Framework for exploring multiple reasoning paths as a tree structure.
- [A Systematic Survey of Prompt Engineering](https://arxiv.org/abs/2402.07927) - Comprehensive survey of 41+ prompt engineering techniques.

**[⬆ back to top](#contents)**

## Resources

*Additional learning materials and community resources.*

### Key Papers

- Chain-of-Thought Prompting (Wei et al., 2022) - Introduced step-by-step reasoning for arithmetic and commonsense tasks.
- ReAct (Yao et al., 2023) - Synergized reasoning with external actions, achieving 91% on HumanEval.
- Toolformer (Schick et al., 2023) - Demonstrated LLMs can teach themselves to use tools via self-supervised learning.
- Self-Consistency (Wang et al., 2022) - Boosted CoT performance by sampling multiple reasoning paths.
- Constitutional AI (Bai et al., 2022) - Alignment breakthrough for training helpful, harmless, and honest AI assistants.

**[⬆ back to top](#contents)**

## Contributing

Your contributions are always welcome! Please submit a pull request or create an issue to add new tools or improve existing entries.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)
