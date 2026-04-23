// mini-llm.js — MiniLLM: assembles modern LLM components
// Uses: RoPE + GQA + SwiGLU + RMSNorm + KV Cache
// This is a simplified LLaMA-like architecture.

import { Matrix } from './matrix.js';
import { precomputeFreqs, applyRoPE } from './rope.js';
import { groupedQueryAttention } from './gqa.js';
import { SwiGLU } from './swiglu.js';
import { topPSample } from './top-p-sampling.js';

export class RMSNorm {
  constructor(dim, eps = 1e-6) {
    this.dim = dim;
    this.eps = eps;
    this.weight = new Float64Array(dim).fill(1);
  }

  forward(x) {
    const out = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++) {
      let sumSq = 0;
      for (let j = 0; j < x.cols; j++) sumSq += x.get(i, j) ** 2;
      const rms = Math.sqrt(sumSq / x.cols + this.eps);
      for (let j = 0; j < x.cols; j++) {
        out.set(i, j, x.get(i, j) / rms * this.weight[j]);
      }
    }
    return out;
  }
}

class TransformerBlock {
  constructor(dModel, nHeads, nKVHeads, dFF) {
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.nKVHeads = nKVHeads;
    
    const headDim = dModel / nHeads;
    const dKV = nKVHeads * headDim;
    
    // Attention projections
    this.Wq = Matrix.random(dModel, dModel).map(v => v * Math.sqrt(2.0 / dModel));
    this.Wk = Matrix.random(dModel, dKV).map(v => v * Math.sqrt(2.0 / dModel));
    this.Wv = Matrix.random(dModel, dKV).map(v => v * Math.sqrt(2.0 / dModel));
    this.Wo = Matrix.random(dModel, dModel).map(v => v * Math.sqrt(2.0 / dModel));
    
    // Layer norms
    this.attnNorm = new RMSNorm(dModel);
    this.ffnNorm = new RMSNorm(dModel);
    
    // SwiGLU FFN
    this.ffn = new SwiGLU(dModel, dFF);
  }

  forward(x, freqs) {
    const seqLen = x.rows;
    const dModel = x.cols;
    const headDim = dModel / this.nHeads;
    const dKV = this.nKVHeads * headDim;
    
    // Pre-norm
    const normed = this.attnNorm.forward(x);
    
    // Q, K, V projections
    const Q = matmul(normed, this.Wq); // seqLen × dModel
    const K = matmul(normed, this.Wk); // seqLen × dKV
    const V = matmul(normed, this.Wv); // seqLen × dKV
    
    // Apply RoPE to Q and K
    const qRoped = applyRoPE(Q, freqs.cos, freqs.sin);
    const kRoped = applyRoPE(K, freqs.cos, freqs.sin);
    
    // GQA
    const attnOut = groupedQueryAttention(qRoped, kRoped, V, this.nHeads, this.nKVHeads, true);
    
    // Output projection + residual
    const projected = matmul(attnOut, this.Wo);
    const residual1 = addMatrices(x, projected);
    
    // FFN with pre-norm + residual
    const ffnNormed = this.ffnNorm.forward(residual1);
    const ffnOut = this.ffn.forward(ffnNormed);
    const residual2 = addMatrices(residual1, ffnOut);
    
    return residual2;
  }
}

export class MiniLLM {
  /**
   * @param {object} config
   * @param {number} config.vocabSize - Vocabulary size
   * @param {number} config.dModel - Model dimension
   * @param {number} config.nHeads - Number of attention heads
   * @param {number} config.nKVHeads - Number of KV heads (for GQA)
   * @param {number} config.nLayers - Number of transformer layers
   * @param {number} config.dFF - FFN hidden dimension
   * @param {number} config.maxLen - Maximum sequence length
   */
  constructor({ vocabSize, dModel = 64, nHeads = 4, nKVHeads = 2, nLayers = 2, dFF = 128, maxLen = 128 }) {
    this.config = { vocabSize, dModel, nHeads, nKVHeads, nLayers, dFF, maxLen };
    
    // Token embedding
    this.embedding = Matrix.random(vocabSize, dModel).map(v => v * 0.02);
    
    // RoPE frequencies
    this.freqs = precomputeFreqs(dModel, maxLen);
    
    // Transformer blocks
    this.blocks = [];
    for (let i = 0; i < nLayers; i++) {
      this.blocks.push(new TransformerBlock(dModel, nHeads, nKVHeads, dFF));
    }
    
    // Final norm + output head
    this.finalNorm = new RMSNorm(dModel);
    this.lmHead = Matrix.random(dModel, vocabSize).map(v => v * 0.02);
  }

  /**
   * Forward pass: tokens → logits.
   * @param {Array<number>} tokens - Input token IDs
   * @returns {Matrix} Logits (seqLen × vocabSize)
   */
  forward(tokens) {
    const seqLen = tokens.length;
    const dModel = this.config.dModel;
    
    // Embed tokens
    let x = new Matrix(seqLen, dModel);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < dModel; j++) {
        x.set(i, j, this.embedding.get(tokens[i], j));
      }
    }
    
    // Forward through transformer blocks
    for (const block of this.blocks) {
      x = block.forward(x, this.freqs);
    }
    
    // Final norm
    x = this.finalNorm.forward(x);
    
    // LM head: x → logits
    const logits = matmul(x, this.lmHead); // seqLen × vocabSize
    
    return logits;
  }

  /**
   * Generate tokens autoregressively.
   * @param {Array<number>} prompt - Initial tokens
   * @param {number} maxNewTokens - Maximum tokens to generate
   * @param {number} temperature - Sampling temperature
   * @param {number} topP - Top-p sampling threshold
   * @returns {Array<number>} Generated token sequence (including prompt)
   */
  generate(prompt, maxNewTokens = 32, temperature = 1.0, topP = 0.9) {
    const tokens = [...prompt];
    
    for (let i = 0; i < maxNewTokens; i++) {
      const logits = this.forward(tokens);
      const lastLogits = new Float64Array(this.config.vocabSize);
      for (let j = 0; j < this.config.vocabSize; j++) {
        lastLogits[j] = logits.get(logits.rows - 1, j);
      }
      
      const nextToken = topPSample(lastLogits, topP, temperature);
      tokens.push(nextToken);
    }
    
    return tokens;
  }

  /**
   * Count total parameters.
   */
  paramCount() {
    const { vocabSize, dModel, nHeads, nKVHeads, nLayers, dFF } = this.config;
    const headDim = dModel / nHeads;
    const dKV = nKVHeads * headDim;
    
    let count = vocabSize * dModel; // embedding
    count += nLayers * (dModel * dModel + dModel * dKV * 2 + dModel * dModel); // Q, K, V, O
    count += nLayers * (new SwiGLU(dModel, dFF)).paramCount(); // FFN
    count += dModel * vocabSize; // lm_head
    return count;
  }
}

// --- Helper functions ---

function matmul(A, B) {
  const result = new Matrix(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < B.cols; j++) {
      let sum = 0;
      for (let k = 0; k < A.cols; k++) {
        sum += A.get(i, k) * B.get(k, j);
      }
      result.set(i, j, sum);
    }
  }
  return result;
}

function addMatrices(A, B) {
  const result = new Matrix(A.rows, A.cols);
  for (let i = 0; i < A.data.length; i++) {
    result.data[i] = A.data[i] + B.data[i];
  }
  return result;
}
