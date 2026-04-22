// llama.js — Llama-style Architecture Components Index
//
// All the building blocks for a Llama 2/3 / Mistral style LLM:
//
// Attention:
//   - FlashAttention (flash-attention.js) — O(N) memory attention
//   - MultiHeadFlashAttention (multi-head-flash-attention.js) — multi-head wrapper
//   - GroupedQueryAttention (grouped-query-attention.js) — fewer KV heads (Llama)
//   - GQA with RoPE (gqa-attention.js) — GQA + rotary embeddings
//
// Position Encoding:
//   - RoPE (rope.js) — rotary position embeddings
//
// Normalization:
//   - RMSNorm (modern-decoder.js) — faster than LayerNorm
//
// FFN:
//   - SwiGLU (modern-decoder.js) — gated linear unit with SiLU
//
// Full Blocks:
//   - ModernDecoderBlock (modern-decoder.js) — complete Llama-style decoder block
//
// Inference:
//   - KVCache (kv-cache.js) — key-value cache for autoregressive generation
//
// Training:
//   - AdamW (adamw.js) — standard LLM optimizer
//   - LR Schedulers (lr-scheduler.js) — warmup, cosine, one-cycle
//   - ScheduledOptimizer (optimizer.js) — optimizer + scheduler integration
//
// This file provides a convenient single-import point.

// Attention
export { FlashAttention, flashAttention, standardAttention } from './flash-attention.js';
export { MultiHeadFlashAttention } from './multi-head-flash-attention.js';
export { GroupedQueryAttention } from './grouped-query-attention.js';

// Position
export { precomputeFreqs, applyRoPE, applyInverseRoPE } from './rope.js';

// Normalization & FFN
export { RMSNorm, SwiGLUFFN, ModernDecoderBlock } from './modern-decoder.js';

// Inference
export { KVCache, ModelKVCache } from './kv-cache.js';

// Training
export { AdamW } from './adamw.js';
export {
  ConstantLR, CosineDecay, LinearWarmup, WarmupScheduler,
  OneCycle, StepDecay, ExponentialDecay
} from './lr-scheduler.js';
export { ScheduledOptimizer } from './optimizer.js';

// Core
export { Matrix } from './matrix.js';
