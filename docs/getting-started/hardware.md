---
title: Hardware Guide
description: Hardware specifications, OS recommendations, and local-vs-cloud trade-offs for running OpenJarvis
search:
  boost: 3
---

# Hardware Guide

OpenJarvis is a local-first AI framework: inference, memory, scheduling, and agentic logic all run on your machine.
The hardware you choose directly determines which models you can run, how fast they respond, and how many parallel
agent pipelines you can sustain.

This guide covers:

1. [What hardware to prioritise](#what-hardware-to-prioritise)
2. [Tier-by-tier build recommendations](#build-tiers)
3. [Which OS to choose](#operating-system)
4. [Local LLM vs cloud LLM — hardware trade-offs](#local-llm-vs-cloud-llm)

---

## What Hardware to Prioritise

### 1. GPU — the primary bottleneck

**GPU VRAM is the single most important specification.** Every LLM inference engine supported by OpenJarvis
(Ollama, vLLM, llama.cpp, SGLang, MLX) keeps the model in GPU memory during inference.  If the model does not fit,
the engine falls back to CPU or offloads layers to system RAM — both of which reduce tokens/second by an order of
magnitude.

**VRAM sizing rule (4-bit quantisation):**

```
VRAM (GB) ≈ model_parameters (B) × 0.5
```

| Model size | Q4 VRAM needed | Example models |
|-----------|----------------|----------------|
| 3 B       | ~2 GB          | Qwen3-3B, Llama-3.2-3B |
| 8 B       | ~5 GB          | Qwen3-8B, Llama-3.1-8B |
| 14 B      | ~9 GB          | DeepSeek-Coder-14B |
| 32 B      | ~18 GB         | Qwen3-32B, Mistral-Small |
| 70 B      | ~38 GB         | Llama-3.3-70B |
| 235 B MoE | ~130 GB        | Qwen3-235B-A22B |

!!! tip "Context window adds VRAM"
    Increasing the KV-cache context (e.g., 32 K → 128 K tokens) can add several gigabytes of VRAM on top of the
    model weight cost.  The OpenJarvis `OrchestratorAgent` and `MonitorOperativeAgent` can consume long contexts
    during multi-turn tool-calling loops — size accordingly.

**Recommended NVIDIA GPUs (2025/2026):**

| Tier | GPU | VRAM | Best for |
|------|-----|------|---------|
| Entry | RTX 4060 Ti | 16 GB | 3 B – 8 B models |
| Midrange | RTX 4070 Ti Super / RX 7900 XTX | 16 – 24 GB | up to 14 B (32 B with offload) |
| High-end | RTX 4090 / RTX 5090 | 24 – 32 GB | up to 32 B locally; 70 B with offload |
| Workstation | RTX 6000 Ada | 48 GB | up to 70 B in full precision |
| Data-centre | A100 80 GB / H100 80 GB | 80 GB | 70 B fp16, multi-agent serving |

!!! note "Apple Silicon"
    Apple M3/M4 chips use **unified memory** shared between CPU and GPU.  A Mac Studio M3 Ultra with 192 GB of
    unified memory can run a full 70 B model at acceptable speed entirely in memory — no discrete GPU required.
    OpenJarvis ships first-class MLX support (`engine: apple_mlx`) for this path.  The trade-off: unified memory
    bandwidth (~800 GB/s on M3 Ultra) is lower than an A100's HBM2e (2 TB/s), so raw tokens/second is lower.

---

### 2. System RAM

System RAM matters when:

- The model (or KV cache) does not fit in VRAM and layers must be **CPU-offloaded** via llama.cpp
- You run **multiple agents concurrently** (each maintains its own context window)
- You use **vector memory backends** (FAISS, ColBERTv2) that load embedding indexes into RAM
- You perform **fine-tuning or GRPO training** via the learning module

| Use case | Recommended RAM |
|----------|-----------------|
| Single lightweight agent (≤ 8 B model, fits in VRAM) | 16 GB |
| Single mid-size agent (8 B – 32 B) | 32 GB |
| Multiple concurrent agents or long-context pipelines | 64 GB |
| 70 B+ models with CPU offload, or fine-tuning | 128 – 256 GB |
| Research workloads (multi-model benchmarks, traces) | 256 GB+ |

Use **DDR5** where your platform supports it — higher bandwidth reduces bottlenecks when the CPU must handle
tokenisation, embedding lookups, and the Python orchestration layer simultaneously.

---

### 3. CPU

The CPU is rarely the inference bottleneck when a GPU is present, but it affects:

- **Tokenisation and preprocessing** — handled in Python before tokens reach the GPU
- **Tool execution** — shell commands, code interpretation, web search, database queries all run on the CPU
- **llama.cpp CPU inference** — if you run without a GPU, the CPU *is* the bottleneck; maximise cache and core count

**Recommended CPUs:**

| Use case | CPU |
|----------|-----|
| Entry / budget | AMD Ryzen 5 7600X, Intel Core i5-14600K |
| Developer workstation | AMD Ryzen 7 7800X3D, Intel Core i7-14700K |
| Power user / multi-agent | AMD Ryzen 9 7950X3D, Intel Core i9-14900K |
| Heavy CPU inference (no GPU) | AMD Threadripper Pro 7985WX, Intel Xeon W9-3595X |

The 3D V-Cache variants (7800X3D, 7950X3D) are particularly good when CPU-side caching affects the agent
orchestration loop or you run llama.cpp without GPU acceleration.

---

### 4. Storage

Model files are large.  Slow storage directly increases cold-start time (loading a 40 GB model from disk).

| Recommendation | Why |
|----------------|-----|
| **NVMe PCIe Gen 4 or Gen 5 SSD** | Sequential read speeds of 5–14 GB/s; a 40 GB model loads in under 10 s |
| **2 TB minimum for a working set of models** | A typical local stack (3 B + 8 B + 14 B + 32 B) consumes ~40 GB |
| Separate fast drive for model storage | Isolates model I/O from OS writes |

Avoid loading models from spinning HDDs or network shares — latency is prohibitive.

---

### 5. Networking and Power

- **Networking** — Not required for local inference. Needed only for cloud API fallback, web search tools
  (Tavily), and channel integrations (Slack, email, etc.). A reliable 100 Mbps+ connection is sufficient.
- **Power supply** — A high-end NVIDIA GPU (RTX 4090: 450 W, RTX 5090: 575 W) combined with a workstation CPU can
  draw 600 – 900 W under sustained inference load. Size your PSU accordingly (add 20 – 30 % headroom).

---

## Build Tiers

### Tier 1 — Entry (up to 8 B models, basic agentic workflows)

| Component | Recommendation |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 Ti 16 GB |
| CPU | AMD Ryzen 5 7600X |
| RAM | 32 GB DDR5 |
| Storage | 1 TB NVMe PCIe Gen 4 |
| OS | Ubuntu 24.04 LTS |
| **Best engine** | **Ollama** |

Runs Qwen3-8B or Llama-3.1-8B comfortably.  Handles `SimpleAgent`, `OrchestratorAgent`, and most built-in tools.

---

### Tier 2 — Developer (up to 32 B models, advanced agentic workflows)

| Component | Recommendation |
|-----------|---------------|
| GPU | NVIDIA RTX 4090 24 GB or AMD RX 7900 XTX 24 GB |
| CPU | AMD Ryzen 9 7950X3D |
| RAM | 64 GB DDR5 |
| Storage | 2 TB NVMe PCIe Gen 4 |
| OS | Ubuntu 24.04 LTS |
| **Best engine** | **Ollama or vLLM** |

Runs Qwen3-32B or DeepSeek-Coder-33B in full Q4.  Supports concurrent agents, long-context pipelines, and
the telemetry + learning stack simultaneously.

---

### Tier 3 — Power User (70 B+ models, multi-agent serving, light fine-tuning)

| Component | Recommendation |
|-----------|---------------|
| GPU | Dual RTX 5090 (2 × 32 GB = 64 GB) or NVIDIA RTX 6000 Ada 48 GB |
| CPU | AMD Threadripper Pro 7985WX or Intel Xeon W9-3595X |
| RAM | 128 – 256 GB DDR5 ECC |
| Storage | 4 TB NVMe PCIe Gen 5 |
| OS | Ubuntu 24.04 LTS or RHEL 9 |
| **Best engine** | **vLLM (multi-GPU tensor parallel)** |

Runs Llama-3.3-70B in Q4 entirely in VRAM.  Supports `MonitorOperativeAgent` long-horizon pipelines, GRPO
training loops, concurrent benchmark suites, and multi-model routers.

---

### Tier 4 — Apple Silicon (macOS-first, unified memory)

| Component | Recommendation |
|-----------|---------------|
| Device | Mac Studio M4 Ultra (192 GB) or Mac Pro M4 Ultra (192 GB) |
| RAM | 192 GB unified memory |
| Storage | 2 TB+ internal SSD |
| OS | macOS 15 Sequoia |
| **Best engine** | **Ollama (MLX backend) or Apple FM shim** |

The unified memory architecture means a 192 GB Mac Studio holds a 70 B model and its KV cache without offloading.
Tokens/second is lower than a data-centre GPU but the experience is silent, power-efficient (≈ 60 W), and
requires zero driver configuration.  All OpenJarvis features work on macOS.

---

## Operating System

### Recommendation: Linux (Ubuntu 24.04 LTS)

Linux is the recommended OS for OpenJarvis for the following reasons:

| Factor | Linux | macOS | Windows |
|--------|-------|-------|---------|
| Inference speed | ⭐⭐⭐⭐⭐ — up to 3× faster than Windows for GPU workloads | ⭐⭐⭐⭐ — excellent on Apple Silicon | ⭐⭐ — OS overhead and driver stack reduce VRAM efficiency |
| CUDA / ROCm support | Native, first-class | Not applicable (no discrete NVIDIA GPUs) | Available but slower; WSL2 rarely matches native |
| Docker / Podman sandbox | Native, zero overhead | Available via Rosetta / OrbStack | WSL2 required; performance overhead |
| Python toolchain (uv, maturin) | Native | Native | Works; slower builds |
| Agentic tool execution (shell, git, code interpreter) | Native | Native | Requires WSL or MSYS2 for full compatibility |
| Long-running services (systemd) | systemd native | launchd (fully supported) | Task Scheduler (limited) |
| GPU driver updates | Stable; CUDA 12.x available day 1 | N/A | Regular driver updates can break builds |
| Recommended distro | Ubuntu 24.04, Fedora 41, RHEL 9 | macOS 14+ | Windows 11 + WSL2 |

**Ubuntu 24.04 LTS** is the primary development and test target for OpenJarvis.  NVIDIA driver installation
is automated (`ubuntu-drivers install`), CUDA 12.x is available from the official NVIDIA APT repository, and
Docker/Podman sandbox support works without additional configuration.

!!! tip "macOS is a first-class platform"
    If you are on Apple Silicon hardware, macOS is an excellent choice.  The OpenJarvis installation script,
    CLI, desktop app, and Ollama MLX backend all work natively.  The `apple_fm_shim` engine provides additional
    optimisations for M-series chips.

!!! warning "Windows users"
    OpenJarvis works on Windows 11 but inference throughput is significantly lower than Linux on identical hardware.
    If you must use Windows, enable **WSL2** (Windows Subsystem for Linux) and run OpenJarvis inside an Ubuntu
    WSL2 instance for near-native performance.  The desktop app's `.exe` installer still works natively.

---

## Local LLM vs Cloud LLM

OpenJarvis supports both inference paths.  The right choice depends on your privacy requirements, budget,
model quality needs, and hardware investment appetite.

### Hardware requirements compared

| Dimension | Local LLM (Ollama / vLLM / llama.cpp) | Cloud LLM (GitHub Copilot / Claude / OpenAI) |
|-----------|---------------------------------------|----------------------------------------------|
| GPU | 16 – 80 GB VRAM (or Apple unified memory) | None — inference runs on provider hardware |
| RAM | 32 – 256 GB depending on model size | Minimal — just enough to run the IDE and OpenJarvis |
| CPU | Modern 8+ core; cache matters for CPU inference | Any modern CPU |
| Storage | 20 GB – 100 GB+ for model files | No model storage needed |
| Network | Optional (tools only) | Required — every inference call leaves your machine |
| Power draw | 100 – 900 W sustained | ~10 – 50 W (no GPU load) |
| Upfront cost | High (GPU hardware) | Low (subscription only) |
| Ongoing cost | Electricity + amortised hardware | $20 – $400 / month per seat (varies by usage) |
| Latency | 0.5 – 3 s (hardware-dependent) | 2 – 10 s (network + provider queue) |
| Privacy | All data stays on your machine | Every prompt is sent to the provider |
| Offline use | Full functionality | Inference unavailable |

### Model quality trade-offs

Local models have closed much of the quality gap with cloud frontier models since 2023:

- **Everyday coding and agentic tasks** — Qwen3-32B, DeepSeek-Coder-33B, and Llama-3.3-70B (all runnable locally
  on Tier 2 – 3 hardware above) perform comparably to GPT-4o on the majority of software-development benchmarks.
- **Complex reasoning, very large context, or novel domains** — Cloud frontier models (Claude 3.5 Sonnet,
  GPT-4o, Gemini 1.5 Pro) still lead, especially for tasks requiring 100 K+ token context windows or
  cutting-edge reasoning.
- **OpenJarvis model router** — The multi-model router can automatically send routine queries to a local model
  and escalate to a cloud model only when confidence is low, giving you the cost and privacy benefits of local
  inference for the majority of queries.

### Agentic software development: specific considerations

| Feature | Local LLM | Cloud LLM |
|---------|-----------|-----------|
| Multi-file code edits | Viable with 32 B+ models | Excellent (Claude 3.5+, GPT-4o) |
| Code execution (ReAct / CodeAct) | Full sandbox (Docker/Podman) | Full sandbox (provider-managed) |
| Private codebase — no data leaves machine | ✅ | ❌ — code sent to provider |
| Context window for large repos | Limited by VRAM (8 K – 128 K) | Up to 200 K tokens (provider-dependent) |
| Concurrent agents | Limited by VRAM | Scales on demand (cost scales too) |
| Fine-tuning on your own traces | ✅ (GRPO / SFT via learning module) | ❌ |
| Availability | Always on (no API outages) | Depends on provider uptime |
| Setup complexity | Moderate (driver install + model pull) | Minimal (API key only) |

### Recommended approach

=== "Privacy-sensitive or air-gapped"

    Use **local LLM exclusively**.  Configure OpenJarvis with Ollama or vLLM, select a model that fits your
    VRAM, and set `router.policy = "local_only"` in `config.toml`.  No data leaves your machine.

=== "Cost-conscious individual developer"

    Use **local LLM for daily work**, cloud API as a fallback for hard problems.  A Tier 2 build (RTX 4090)
    running Qwen3-32B handles ~90 % of everyday development tasks.  Configure the multi-model router:

    ```toml
    [router]
    policy    = "heuristic"
    local_max = "qwen3:32b"
    cloud_fallback = "claude-3-5-sonnet-20241022"
    ```

=== "Team or enterprise"

    Evaluate whether dedicated on-premises hardware (GPU server running vLLM) is more cost-effective than
    per-seat cloud subscriptions.  At scale, a single A100 80 GB server can serve dozens of developers
    concurrently and pays for itself within months compared to per-seat cloud pricing.

=== "Getting started quickly"

    Use **cloud APIs** initially.  Configure an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, set the engine to
    `cloud`, and focus on learning the OpenJarvis agent and tool patterns.  Migrate to a local engine later
    when you know which model size your workload actually needs.

---

## Quick Reference

| Goal | Minimum hardware | Recommended hardware |
|------|-----------------|---------------------|
| Run 3 B – 8 B models (basic chat, simple agents) | 8 GB VRAM GPU, 16 GB RAM | RTX 4060 Ti 16 GB, 32 GB RAM |
| Run 14 B – 32 B models (coding, research agents) | 16 GB VRAM GPU, 32 GB RAM | RTX 4090 24 GB, 64 GB RAM |
| Run 70 B models locally | 40 GB VRAM (dual GPU) or Apple M3/M4 Ultra 192 GB | Dual RTX 5090, 128 GB RAM |
| Multi-agent pipelines, concurrent users | 24 GB VRAM, 64 GB RAM | RTX 4090 or RTX 6000 Ada, 128 GB RAM |
| Fine-tuning / GRPO training | 24 GB VRAM, 64 GB RAM | A100 80 GB, 256 GB RAM |
| Cloud API only (no local inference) | Any modern PC / laptop | Any modern PC / laptop |

!!! info "Let OpenJarvis detect your hardware"
    Run `jarvis init` after installation.  It auto-detects GPU vendor, VRAM, CPU, and available RAM, then writes
    a `config.toml` with the recommended engine, model size, and quantisation level for your specific hardware.

## Further Reading

- [Ollama hardware requirements](https://ollama.com/library) — per-model VRAM estimates
- [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) — GPU prerequisites
- [llama.cpp performance](https://github.com/ggerganov/llama.cpp#performance) — CPU and GPU benchmark tables
- [Apple MLX](https://github.com/ml-explore/mlx) — Apple Silicon inference framework
- [Intelligence Per Watt research](https://www.intelligence-per-watt.ai/) — the efficiency research behind OpenJarvis

---

*Last updated: March 2026.  The local LLM hardware landscape evolves rapidly — model efficiency improves roughly
5× every two years, meaning hardware that struggles today will handle larger models within 12 – 18 months.*
