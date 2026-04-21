// modern-decoder.js — Llama-style Decoder Block
// Architecture: Pre-norm + GQA w/ RoPE + SwiGLU FFN
// This is how modern LLMs (Llama 2/3, Mistral, Gemma) structure each layer.
//
// Key differences from the original Transformer ("Attention Is All You Need"):
// 1. Pre-LayerNorm instead of Post-LayerNorm (training stability)
// 2. RMSNorm instead of LayerNorm (faster, no mean subtraction)
// 3. GQA instead of MHA (less KV memory)
// 4. RoPE instead of sinusoidal/learned position embedding
// 5. SwiGLU instead of ReLU FFN (better performance)
// 6. No bias in linear layers (Llama convention)

import { Matrix } from './matrix.js';
import { GroupedQueryAttention } from './gqa-attention.js';

/**
 * RMSNorm — Root Mean Square Layer Normalization
 * Used in Llama instead of standard LayerNorm.
 * Simpler: no mean subtraction, just normalize by RMS.
 *
 * RMSNorm(x) = x / RMS(x) * γ
 * where RMS(x) = sqrt(mean(x²) + ε)
 */
export class RMSNorm {
  constructor(dim, epsilon = 1e-6) {
    this.dim = dim;
    this.epsilon = epsilon;
    this.gamma = new Float64Array(dim).fill(1.0); // learnable scale
    this.outputSize = dim;
    this._cache = null;
  }

  /**
   * Forward: normalize input per-position.
   * @param {Matrix} input - [batch, seqLen * dim]
   */
  forward(input) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dim);
    const result = new Matrix(batchSize, input.cols);
    this._cache = { input, rms: [] };

    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const offset = t * this.dim;
        let sumSq = 0;
        for (let d = 0; d < this.dim; d++) {
          const v = input.get(b, offset + d);
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / this.dim + this.epsilon);
        this._cache.rms.push(rms);

        for (let d = 0; d < this.dim; d++) {
          const normalized = input.get(b, offset + d) / rms;
          result.set(b, offset + d, normalized * this.gamma[d]);
        }
      }
    }
    return result;
  }
}

/**
 * SwiGLU Feed-Forward Network
 * Used in Llama instead of standard ReLU FFN.
 * FFN_SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
 * where Swish(x) = x * sigmoid(x)
 *
 * This has 3 weight matrices instead of 2 (standard FFN has W1, W2).
 * The hidden dim is typically (2/3 * 4 * dModel), rounded to nearest multiple of 256.
 */
export class SwiGLUFFN {
  constructor(dModel, dHidden = null) {
    this.dModel = dModel;
    // Llama convention: hidden = round_to_multiple(2/3 * 4 * dModel, 256)
    this.dHidden = dHidden || Math.ceil((2 / 3 * 4 * dModel) / 256) * 256 || dModel * 2;
    this.outputSize = dModel;

    const scale = Math.sqrt(2 / (dModel + this.dHidden));
    this.W1 = Matrix.random(dModel, this.dHidden).mul(scale); // gate projection
    this.W3 = Matrix.random(dModel, this.dHidden).mul(scale); // up projection
    this.W2 = Matrix.random(this.dHidden, dModel).mul(scale); // down projection

    this._cache = null;
  }

  /**
   * Swish activation: x * sigmoid(x)
   */
  static swish(x) {
    return x / (1 + Math.exp(-x));
  }

  /**
   * Forward: SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
   * @param {Matrix} input - [numPositions, dModel]
   */
  forward(input) {
    const gate = input.dot(this.W1);  // [n, dHidden]
    const up = input.dot(this.W3);    // [n, dHidden]

    // Apply Swish to gate, then element-wise multiply with up
    const gated = new Matrix(gate.rows, gate.cols);
    for (let r = 0; r < gate.rows; r++) {
      for (let c = 0; c < gate.cols; c++) {
        gated.set(r, c, SwiGLUFFN.swish(gate.get(r, c)) * up.get(r, c));
      }
    }

    this._cache = { input, gate, up, gated };
    return gated.dot(this.W2); // [n, dModel]
  }
}

/**
 * Modern Decoder Block (Llama-style)
 *
 * Architecture:
 *   x → RMSNorm → GQA(RoPE) → +x → RMSNorm → SwiGLU → +x
 *
 * Pre-norm: normalize BEFORE attention/FFN, not after.
 * Two residual connections: around attention and around FFN.
 */
export class ModernDecoderBlock {
  constructor(dModel, numQHeads, numKVHeads, { dHidden, maxSeqLen = 2048 } = {}) {
    this.dModel = dModel;
    this.outputSize = dModel;

    this.attnNorm = new RMSNorm(dModel);
    this.ffnNorm = new RMSNorm(dModel);
    this.attention = new GroupedQueryAttention(dModel, numQHeads, numKVHeads, {
      causal: true,
      useRoPE: true,
      maxSeqLen,
    });
    this.ffn = new SwiGLUFFN(dModel, dHidden);
  }

  /**
   * Forward pass.
   * @param {Matrix} input - [batch, seqLen * dModel]
   * @param {boolean} useCache - enable KV-cache for incremental generation
   */
  forward(input, useCache = false) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);

    // Pre-norm attention with residual
    const normed1 = this.attnNorm.forward(input);
    const attnOut = this.attention.forward(normed1, useCache);
    const residual1 = addMatrices(input, attnOut);

    // Pre-norm FFN with residual
    const normed2 = this.ffnNorm.forward(residual1);

    // FFN operates per-position: reshape to [batch*seqLen, dModel]
    const allPositions = new Matrix(batchSize * seqLen, this.dModel);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        for (let d = 0; d < this.dModel; d++) {
          allPositions.set(b * seqLen + t, d, normed2.get(b, t * this.dModel + d));
        }
      }
    }

    const ffnOut = this.ffn.forward(allPositions);

    // Scatter back
    const ffnScattered = new Matrix(batchSize, input.cols);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        for (let d = 0; d < this.dModel; d++) {
          ffnScattered.set(b, t * this.dModel + d, ffnOut.get(b * seqLen + t, d));
        }
      }
    }

    return addMatrices(residual1, ffnScattered);
  }

  clearCache() {
    this.attention.clearCache();
  }
}

/**
 * Stack of decoder blocks (mini Llama).
 */
export class ModernDecoder {
  constructor(numLayers, dModel, numQHeads, numKVHeads, vocabSize, { dHidden, maxSeqLen = 2048 } = {}) {
    this.dModel = dModel;
    this.vocabSize = vocabSize;
    this.maxSeqLen = maxSeqLen;

    // Token embedding: [vocabSize, dModel]
    const scale = Math.sqrt(2 / (vocabSize + dModel));
    this.embedding = Matrix.random(vocabSize, dModel).mul(scale);

    // Decoder blocks
    this.blocks = [];
    for (let i = 0; i < numLayers; i++) {
      this.blocks.push(new ModernDecoderBlock(dModel, numQHeads, numKVHeads, { dHidden, maxSeqLen }));
    }

    // Final norm + output projection
    this.finalNorm = new RMSNorm(dModel);
    this.outputProj = Matrix.random(dModel, vocabSize).mul(Math.sqrt(2 / (dModel + vocabSize)));
  }

  /**
   * Forward pass: token ids → logits.
   * @param {number[][]} tokenIds - [batch][seqLen] array of token IDs
   * @param {boolean} useCache - enable KV-cache
   * @returns {Matrix} logits [batch, seqLen * vocabSize]
   */
  forward(tokenIds, useCache = false) {
    const batchSize = tokenIds.length;
    const seqLen = tokenIds[0].length;

    // Embed tokens: [batch, seqLen * dModel]
    let hidden = new Matrix(batchSize, seqLen * this.dModel);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const tokenId = tokenIds[b][t];
        for (let d = 0; d < this.dModel; d++) {
          hidden.set(b, t * this.dModel + d, this.embedding.get(tokenId, d));
        }
      }
    }

    // Pass through decoder blocks
    for (const block of this.blocks) {
      hidden = block.forward(hidden, useCache);
    }

    // Final norm
    hidden = this.finalNorm.forward(hidden);

    // Project to vocabulary: [batch, seqLen * vocabSize]
    const logits = new Matrix(batchSize, seqLen * this.vocabSize);
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        // Extract position vector
        const pos = new Matrix(1, this.dModel);
        for (let d = 0; d < this.dModel; d++) {
          pos.set(0, d, hidden.get(b, t * this.dModel + d));
        }
        const tok_logits = pos.dot(this.outputProj);
        for (let v = 0; v < this.vocabSize; v++) {
          logits.set(b, t * this.vocabSize + v, tok_logits.get(0, v));
        }
      }
    }

    return logits;
  }

  /**
   * Generate tokens autoregressively using greedy decoding.
   * @param {number[]} prompt - initial token IDs
   * @param {number} maxNewTokens - max tokens to generate
   * @returns {number[]} generated token IDs (including prompt)
   */
  generate(prompt, maxNewTokens = 20) {
    this.clearCache();
    const tokens = [...prompt];

    // Process prompt (prefill)
    const logits = this.forward([tokens], true);

    // Greedy: pick argmax of last position
    let nextToken = argmaxSlice(logits, 0, (tokens.length - 1) * this.vocabSize, this.vocabSize);
    tokens.push(nextToken);

    // Decode one token at a time
    for (let i = 1; i < maxNewTokens; i++) {
      const stepLogits = this.forward([[nextToken]], true);
      nextToken = argmaxSlice(stepLogits, 0, 0, this.vocabSize);
      tokens.push(nextToken);
    }

    return tokens;
  }

  clearCache() {
    for (const block of this.blocks) block.clearCache();
  }

  /**
   * Parameter count (approximate).
   */
  paramCount() {
    let count = this.embedding.rows * this.embedding.cols; // embedding
    for (const block of this.blocks) {
      const attn = block.attention;
      count += attn.Wq.rows * attn.Wq.cols; // Wq
      count += attn.Wk.rows * attn.Wk.cols; // Wk
      count += attn.Wv.rows * attn.Wv.cols; // Wv
      count += attn.Wo.rows * attn.Wo.cols; // Wo
      count += block.ffn.W1.rows * block.ffn.W1.cols; // FFN gate
      count += block.ffn.W3.rows * block.ffn.W3.cols; // FFN up
      count += block.ffn.W2.rows * block.ffn.W2.cols; // FFN down
      count += block.attnNorm.dim * 2; // norms
      count += block.ffnNorm.dim * 2;
    }
    count += this.outputProj.rows * this.outputProj.cols; // output
    return count;
  }
}

// --- Helpers ---

function addMatrices(a, b) {
  const result = new Matrix(a.rows, a.cols);
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < a.cols; c++)
      result.set(r, c, a.get(r, c) + b.get(r, c));
  return result;
}

function argmaxSlice(mat, row, start, len) {
  let maxVal = -Infinity, maxIdx = 0;
  for (let i = 0; i < len; i++) {
    const v = mat.get(row, start + i);
    if (v > maxVal) { maxVal = v; maxIdx = i; }
  }
  return maxIdx;
}
