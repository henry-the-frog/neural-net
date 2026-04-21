# Modern LLM Architecture — Built From Scratch

A complete implementation of modern Large Language Model components, from tokenization to alignment.

## Components

### Tokenization
- **`bpe.js`** — Byte Pair Encoding tokenizer
  - Training: iterative most-frequent-pair merging
  - Encode/decode roundtrip, special tokens, export/import
  - 13 tests

### Position Encoding
- **`rope.js`** — Rotary Position Embedding (RoPE)
  - Encodes position by rotating Q/K dimension pairs
  - Key property: dot product depends only on relative distance
  - KV-cache offset support for incremental generation
  - 11 tests

### Attention Mechanisms
- **`gqa-attention.js`** — Grouped Query Attention with KV-Cache
  - Configurable Q/KV head ratio: MHA (1:1), GQA (N:M), MQA (N:1)
  - RoPE integration, causal masking
  - KV-cache for O(1) incremental generation
  - 20 tests

- **`flash-attention.js`** — Flash Attention (tiled computation)
  - Online softmax for O(N × tileSize) memory instead of O(N²)
  - Exact match with standard attention across all tile sizes
  - 8x memory reduction demonstrated
  - 8 tests

- **`sliding-window.js`** — Sliding Window Attention (Mistral-style)
  - Each token attends to W nearest tokens: O(N × W)
  - Bounded KV-cache with auto-eviction
  - 10 tests

### Model Architecture
- **`modern-decoder.js`** — Llama-style Decoder Block
  - **RMSNorm**: Faster than LayerNorm (no mean subtraction)
  - **SwiGLU FFN**: 3-matrix gated FFN with Swish activation
  - **Pre-norm** residual connections
  - **ModernDecoder**: Stacked blocks + embedding + generation
  - KV-cache support for autoregressive decoding
  - 16 tests

- **`moe.js`** — Mixture of Experts (Mixtral-style)
  - N experts with learned top-K routing
  - Load balancing loss for training
  - Parameter efficiency: 25% active params with top-2/8
  - 12 tests

### Inference Optimization
- **`sampling.js`** — Token Sampling Strategies
  - Temperature, top-k, top-p (nucleus), repetition penalty
  - Combined pipeline: temperature → top-k → top-p → sample
  - 23 tests

- **`speculative-decoding.js`** — Speculative Decoding
  - Draft model generates K candidates, target verifies in 1 pass
  - Probabilistic acceptance with adjusted rejection sampling
  - Same distribution as target model, fewer forward passes
  - 6 tests

- **`quantization.js`** — Weight Quantization
  - INT8 absmax: ~8x compression, per-tensor
  - INT4 group: ~16x compression, configurable group size
  - 10 tests

- **`kv-cache-compression.js`** — KV-Cache Compression
  - INT8 per-vector quantization of cached K/V
  - >3x memory reduction with MAE < 0.1
  - Eviction support for bounded cache
  - 6 tests

### Training & Fine-Tuning
- **`simple-train.js`** — SPSA Training
  - Gradient-free optimization via parameter perturbation
  - Verified: 58% loss reduction on simple patterns
  - 2 tests

- **`lora.js`** — LoRA (Low-Rank Adaptation)
  - W' = W + (α/r) · BA decomposition
  - 256x parameter compression for rank-8
  - Merge adapter into base weight for zero overhead
  - Export/import for adapter swapping
  - 8 tests

### Alignment
- **`dpo.js`** — Direct Preference Optimization
  - Simpler alternative to RLHF
  - Loss: -log σ(β · (chosen_ratio - rejected_ratio))
  - Implicit reward extraction from trained model
  - 7 tests

## End-to-End Pipeline
**`llm-demo.test.js`** wires everything together:
```
Text → BPE Tokenizer → Token IDs → ModernDecoder → Logits → Sampling → Token IDs → Decode → Text
```
- Benchmarked: ~4000 tokens/second on mini model
- 6 integration tests

## Test Count
- **158 tests** across 15 new test files
- All components independently tested and verified

## Architecture Diagram
```
┌─────────────┐
│ BPE Tokenizer│ text → tokens → text
└──────┬──────┘
       │
┌──────▼──────┐
│  Embedding   │ token IDs → vectors
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│  Decoder Block (×N layers)       │
│  ┌────────────────────────────┐  │
│  │ RMSNorm → GQA+RoPE → +res │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │ RMSNorm → SwiGLU/MoE → +res│  │
│  └────────────────────────────┘  │
└──────┬──────────────────────────┘
       │
┌──────▼──────┐
│  RMSNorm     │ final normalization
└──────┬──────┘
       │
┌──────▼──────┐     ┌──────────────┐
│ Output Proj  │────►│  Sampling     │ temp, top-k/p
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  BPE Decode   │ tokens → text
                    └──────────────┘
```

## References
- Llama 2/3 (Meta): GQA, RoPE, SwiGLU, RMSNorm
- Mistral: Sliding Window Attention
- Flash Attention (Dao et al., 2022): Tiled attention
- Mixtral: Mixture of Experts
- LoRA (Hu et al., 2021): Low-Rank Adaptation
- DPO (Rafailov et al., 2023): Direct Preference Optimization
- GPT-2: BPE tokenizer
