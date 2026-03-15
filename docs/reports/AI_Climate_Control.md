# AI Computational Climate Control

As Artificial Intelligence scales, the fundamental limits of computation are no longer just about parameter counts or dataset sizes. The limits are mathematical and physical. At the lowest level of execution, modern AI pipelines are subjected to extreme structural stresses created by floating-point arithmetic and memory geometry. 

We call this the **Computational Climate**.

To study this climate without being drowned in the distributed scaffolding of massive server farms, we observe it through the micro-cosmos of **Bolmo-1B**. Because of its unique architecture featuring byte-level parsing, dynamic patch boundaries, and recurrent local encoders alongside global transformers, Bolmo contains the complete AI stack within a single model. It allows us to map the three primary domains of AI computation directly to the existential "weather events" that choke standard processors.

The following defines the physical reality of the AI computational climate across its three core domains.

---

## Domain 1: Networks (Inference Engine)
*The heart of the LLM. This domain pumps the tokens and drives the forward pass. In Bolmo, this is the global `BolmoDecoderLayer` and its self-attention mechanism.*

### Transcendental Frost (Exponential: `exp`)
The frozen hell inside every softmax and activation. In Bolmo's scaled dot-product attention, after the raw scores are calculated, they must be normalized into probabilities. This requires calculating `exp(x)` for billions of values per second. Transcendentals are not native, single-cycle operations; they are iterative approximations. When a GPU hits a softmax layer, the arithmetic pipeline plunges into a deep freeze.

### Division Permafrost (Reciprocal: `1/x`)
The permanent ice layer that never melts. Division is the most deceptive bottleneck in computing. It looks like one operation, but it costs orders of magnitude more than a multiply. In Bolmo, this permafrost hides inside every `BolmoRMSNorm` layer, which calculates variances and scales the outputs, and inside the attention scaling factor. It is the inescapable latency floor of every transformer layer.

---

## Domain 2: Databases (Retrieval & RAG)
*The memory of AI. This domain handles context fetching and historical reference. In Bolmo, this is not an external vector database, but the internal KV Cache where historical keys and values are stored and searched during autoregressive decoding.*

### Similarity Tsunami (Dot Product)
A wall of multiply-adds that drowns every vector search. When a model generates a token, its Query vector must search against the entire historical context. In Bolmo, this is the `Q @ K^T` operation spanning up to 65,536 positional embeddings. The arithmetic is simple, but the sheer volume of data movement floods the memory bandwidth, defining the thermal and physical limits of inference speed.

### Argmax Drought (Comparison & Selection)
The desert where parallelism dies and only one token survives. After the tsunami of dot products computes the similarities, the system must decide which context actually matters. Sorting, Top-K selection, and max-pooling force the hardware to halt its massive parallel processing and execute serial comparison chains. 

---

## Domain 3: Applications (Agents & Routing)
*The brain of AI. This domain governs logic, tool use, and dynamic decision-making. In Bolmo, this agentic routing is built directly into the model via the `BolmoBoundaryPredictor`, which decides dynamically where to segment raw bytes into semantic patches.*

### Branch Fog (Conditional Selection)
The thick fog that blinds the processor and triggers pipeline flushes. Modern AI relies increasingly on Mixture of Experts (MoE), dynamic routing, and agentic tool calls. Hardware relies on branch prediction to keep its pipelines full. When an AI agent must stop, evaluate a condition, and route to a specific tool or expert, the hardware cannot predict the path. The pipeline flushes, speculative execution collapses, and throughput drops to a fraction of its potential.

### Distance Freeze (Square Root: `sqrt`)
The sudden hard freeze that hits every time we need true L2 distance or magnitude normalization. Inside Bolmo's boundary predictor, representations are normalized via cosine similarity to decide if a boundary should be drawn. This requires computing square roots over high-dimensional vectors. Like division, the square root halts the fast flow of linear algebra, forcing the hardware to wait for complex, multi-cycle approximations before a routing decision can be made.

---

## Summary

Every complete AI architecture lives through this cycle. It generates attention (The Frost and Permafrost), searches its context (The Tsunami and Drought), and routes its logic (The Fog and Freeze). 

By defining the computational climate in these exact mathematical terms, we isolate the specific structural bottlenecks that standard floating-point hardware cannot efficiently bypass. Bolmo provides the perfect observable environment to witness these events in real time.

---

## Bolmo Operational Control Surfaces

The current Gyroscopic climate-control implementation operationalises the six climate events at two exact intervention surfaces inside Bolmo:

### 1. Encode-side control surface: boundary law
Bolmo's non-causal `BolmoBoundaryPredictor` is the root geometric control surface of the model. It determines where byte streams are segmented into latent patches, and therefore directly controls:

- patch count
- bytes per patch
- global attention sequence length
- KV cache growth
- local/global compute allocation

In the current aQPU integration, GyroLabe replaces the boundary law from a purely cosine-distance-driven field with a structurally controlled field over exact byte-native observables. This directly targets:

- **Distance Freeze**
- **Division Permafrost**

and indirectly targets:

- **Similarity Tsunami**
- **Transcendental Frost**

through patch-geometry coarsening.

### 2. Decode-side control surface: fused content/phase selection
Bolmo's 512-way output alphabet is not a flat content vocabulary. It is a fused transducer alphabet consisting of:

- byte content
- plus a boundary phase bit

In the current aQPU integration, GyroGraph replaces flat 512-way decode selection with paired content/phase quotient selection plus temporal phase hysteresis. This directly targets:

- **Argmax Drought**
- **Branch Fog**

by reducing flat byte-phase competition and stabilising the emitted boundary phase across decode steps.

A mechanistic instrumentation result also establishes that fused boundary token IDs are not fed back into the byte embedding layer during generation. This means the fused Bolmo alphabet is operationally a decode-phase output algebra rather than a recurrent runtime input embedding alphabet. Accordingly, the Gyroscopic integration treats fused boundary phase as a decode-side control variable, not as an encode-side byte-embedding phenomenon.

### Current validated proxies

The current implementation validates the following model-native climate proxies:

- **Patch count** and **mean bytes per patch** as direct geometry / compute-allocation observables
- **Attention proxy** = `patch_count^2` as a prefill/global similarity burden proxy
- **KV proxy** = `patch_count` as a memory-growth proxy
- **Gauge flip rate** as a decode-phase turbulence proxy
- **Phase redundancy** = `raw_support_count_mean - support_count_mean` as a fused-vs-quotiented decode competition proxy

These proxies are not external runtime counters. They are extracted from the exact byte-to-patch and content-to-phase control surfaces native to Bolmo.