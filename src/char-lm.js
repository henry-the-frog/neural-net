// char-lm.js — Tiny character-level language model
// Uses a minimal decoder-only transformer (GPT-style) to learn character sequences.
// Zero dependencies beyond matrix.js and the existing layer infrastructure.

import { Matrix } from './matrix.js';
import { AdamW } from './adamw.js';
import { CosineAnnealingLR } from './lr-scheduler.js';

/**
 * Character-level tokenizer — maps chars to integer IDs
 */
export class CharTokenizer {
  constructor() {
    this.charToId = new Map();
    this.idToChar = new Map();
    this.vocabSize = 0;
  }

  fit(text) {
    const chars = new Set(text);
    const sorted = [...chars].sort();
    this.charToId.clear();
    this.idToChar.clear();
    sorted.forEach((ch, i) => {
      this.charToId.set(ch, i);
      this.idToChar.set(i, ch);
    });
    this.vocabSize = sorted.length;
    return this;
  }

  encode(text) {
    return [...text].map(ch => this.charToId.get(ch) ?? 0);
  }

  decode(ids) {
    return ids.map(id => this.idToChar.get(id) ?? '?').join('');
  }
}

/**
 * Causal self-attention — each position can only attend to earlier positions.
 * Implements scaled dot-product attention with a causal mask.
 */
class CausalSelfAttention {
  constructor(dModel, nHeads) {
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.headDim = Math.floor(dModel / nHeads);
    
    // Q, K, V projections + output projection
    const scale = Math.sqrt(2 / dModel);
    this.Wq = Matrix.random(dModel, dModel, scale);
    this.Wk = Matrix.random(dModel, dModel, scale);
    this.Wv = Matrix.random(dModel, dModel, scale);
    this.Wo = Matrix.random(dModel, dModel, scale);
    
    // Gradients
    this.dWq = new Matrix(dModel, dModel);
    this.dWk = new Matrix(dModel, dModel);
    this.dWv = new Matrix(dModel, dModel);
    this.dWo = new Matrix(dModel, dModel);
  }

  forward(x, seqLen) {
    // x: [seqLen, dModel]
    this._input = x;
    this._seqLen = seqLen;
    
    // Compute Q, K, V
    const Q = matmul(x, this.Wq);
    const K = matmul(x, this.Wk);
    const V = matmul(x, this.Wv);
    this._Q = Q; this._K = K; this._V = V;
    
    // Scaled dot-product attention with causal mask
    const scale = Math.sqrt(this.headDim);
    const scores = matmul(Q, transpose(K));
    
    // Apply causal mask (lower triangular) + scaling
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        if (j > i) {
          scores.set(i, j, -1e9); // mask future positions
        } else {
          scores.set(i, j, scores.get(i, j) / scale);
        }
      }
    }
    
    // Softmax per row
    const attnWeights = softmaxRows(scores);
    this._attnWeights = attnWeights;
    
    // Weighted sum of values
    const attnOutput = matmul(attnWeights, V);
    
    // Output projection
    const output = matmul(attnOutput, this.Wo);
    this._attnOutput = attnOutput;
    return output;
  }

  backward(dOutput) {
    const seqLen = this._seqLen;
    
    // dOutput: [seqLen, dModel]
    // backward through Wo
    const dAttnOutput = matmul(dOutput, transpose(this.Wo));
    this.dWo = matmul(transpose(this._attnOutput), dOutput);
    
    // backward through attention: d(softmax(scores) @ V)
    const dV = matmul(transpose(this._attnWeights), dAttnOutput);
    const dAttnWeights = matmul(dAttnOutput, transpose(this._V));
    
    // backward through softmax
    const dScores = softmaxBackward(this._attnWeights, dAttnWeights);
    
    // Apply causal mask to scores gradient
    const scale = Math.sqrt(this.headDim);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        if (j > i) dScores.set(i, j, 0);
        else dScores.set(i, j, dScores.get(i, j) / scale);
      }
    }
    
    // backward through QK^T
    const dQ = matmul(dScores, this._K);
    const dK = matmul(transpose(dScores), this._Q);
    
    // Accumulate weight gradients
    this.dWq = matmul(transpose(this._input), dQ);
    this.dWk = matmul(transpose(this._input), dK);
    this.dWv = matmul(transpose(this._input), dV);
    
    // Input gradient
    const dX = matmul(dQ, transpose(this.Wq));
    addInPlace(dX, matmul(dK, transpose(this.Wk)));
    addInPlace(dX, matmul(dV, transpose(this.Wv)));
    
    return dX;
  }

  update(lr) {
    updateWeight(this.Wq, this.dWq, lr);
    updateWeight(this.Wk, this.dWk, lr);
    updateWeight(this.Wv, this.dWv, lr);
    updateWeight(this.Wo, this.dWo, lr);
  }
}

/**
 * Simple feedforward block (MLP) used in transformer blocks
 */
class FFN {
  constructor(dModel, dFF) {
    const scale1 = Math.sqrt(2 / dModel);
    const scale2 = Math.sqrt(2 / dFF);
    this.W1 = Matrix.random(dModel, dFF, scale1);
    this.b1 = new Matrix(1, dFF);
    this.W2 = Matrix.random(dFF, dModel, scale2);
    this.b2 = new Matrix(1, dModel);
    
    this.dW1 = new Matrix(dModel, dFF);
    this.db1 = new Matrix(1, dFF);
    this.dW2 = new Matrix(dFF, dModel);
    this.db2 = new Matrix(1, dModel);
  }

  forward(x) {
    this._input = x;
    // x: [seqLen, dModel]
    const h = matmul(x, this.W1);
    addBias(h, this.b1);
    this._preRelu = h;
    relu(h);
    this._hidden = h;
    const out = matmul(h, this.W2);
    addBias(out, this.b2);
    return out;
  }

  backward(dOutput) {
    // dOutput: [seqLen, dModel]
    this.dW2 = matmul(transpose(this._hidden), dOutput);
    this.db2 = sumRows(dOutput);
    
    let dH = matmul(dOutput, transpose(this.W2));
    // ReLU backward
    for (let i = 0; i < dH.rows; i++) {
      for (let j = 0; j < dH.cols; j++) {
        if (this._preRelu.get(i, j) <= 0) dH.set(i, j, 0);
      }
    }
    
    this.dW1 = matmul(transpose(this._input), dH);
    this.db1 = sumRows(dH);
    
    return matmul(dH, transpose(this.W1));
  }

  update(lr) {
    updateWeight(this.W1, this.dW1, lr);
    updateWeight(this.b1, this.db1, lr);
    updateWeight(this.W2, this.dW2, lr);
    updateWeight(this.b2, this.db2, lr);
  }
}

/**
 * Decoder block: causal attention + FFN with residual connections
 */
class DecoderBlock {
  constructor(dModel, nHeads, dFF) {
    this.attn = new CausalSelfAttention(dModel, nHeads);
    this.ffn = new FFN(dModel, dFF);
  }

  forward(x, seqLen) {
    // Residual + attention
    const attnOut = this.attn.forward(x, seqLen);
    const x2 = add(x, attnOut);
    
    // Residual + FFN
    const ffnOut = this.ffn.forward(x2);
    return add(x2, ffnOut);
  }

  backward(dOutput) {
    // FFN residual backward
    const dFFN = this.ffn.backward(dOutput);
    const dX2 = add(dOutput, dFFN);
    
    // Attention residual backward
    const dAttn = this.attn.backward(dX2);
    return add(dX2, dAttn);
  }

  update(lr) {
    this.attn.update(lr);
    this.ffn.update(lr);
  }
}

/**
 * CharLM — Character-level language model (decoder-only transformer)
 * 
 * Architecture:
 * - Token embedding: [vocabSize, dModel]
 * - Positional encoding (sinusoidal)
 * - N decoder blocks (causal attention + FFN)
 * - Output projection: [dModel, vocabSize]
 * - Softmax + cross-entropy loss
 */
export class CharLM {
  constructor({ vocabSize, dModel = 32, nHeads = 4, nLayers = 2, dFF = 64, maxLen = 64 }) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.maxLen = maxLen;
    
    // Token embedding
    const embScale = Math.sqrt(1 / dModel);
    this.embedding = Matrix.random(vocabSize, dModel, embScale);
    this.dEmbedding = new Matrix(vocabSize, dModel);
    
    // Positional encoding (sinusoidal, non-learnable)
    this.pe = new Matrix(maxLen, dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        this.pe.set(pos, i, i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
      }
    }
    
    // Decoder blocks
    this.blocks = [];
    for (let i = 0; i < nLayers; i++) {
      this.blocks.push(new DecoderBlock(dModel, nHeads, dFF));
    }
    
    // Output projection to vocab
    const outScale = Math.sqrt(2 / dModel);
    this.outputW = Matrix.random(dModel, vocabSize, outScale);
    this.outputB = new Matrix(1, vocabSize);
    this.dOutputW = new Matrix(dModel, vocabSize);
    this.dOutputB = new Matrix(1, vocabSize);
  }

  /**
   * Forward pass: tokens → logits
   * @param {number[]} tokens - Array of token IDs
   * @returns {Matrix} logits [seqLen, vocabSize]
   */
  forward(tokens) {
    const seqLen = tokens.length;
    
    // Lookup embeddings + add positional encoding
    const x = new Matrix(seqLen, this.dModel);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < this.dModel; d++) {
        x.set(t, d, this.embedding.get(tokens[t], d) + this.pe.get(t, d));
      }
    }
    this._tokens = tokens;
    this._embInput = x;
    
    // Pass through decoder blocks
    let h = x;
    for (const block of this.blocks) {
      h = block.forward(h, seqLen);
    }
    this._finalHidden = h;
    
    // Output projection → logits
    const logits = matmul(h, this.outputW);
    addBias(logits, this.outputB);
    
    return logits;
  }

  /**
   * Compute cross-entropy loss
   * @param {Matrix} logits [seqLen, vocabSize]
   * @param {number[]} targets - Target token IDs for each position
   * @returns {{ loss: number, dLogits: Matrix }}
   */
  loss(logits, targets) {
    const seqLen = logits.rows;
    const probs = softmaxRows(logits);
    this._probs = probs;
    
    // Cross-entropy loss: -sum(log(prob[target])) / seqLen
    let totalLoss = 0;
    for (let t = 0; t < seqLen; t++) {
      const p = Math.max(probs.get(t, targets[t]), 1e-10);
      totalLoss -= Math.log(p);
    }
    totalLoss /= seqLen;
    
    // Gradient of softmax + cross-entropy
    const dLogits = new Matrix(seqLen, this.vocabSize);
    for (let t = 0; t < seqLen; t++) {
      for (let v = 0; v < this.vocabSize; v++) {
        dLogits.set(t, v, probs.get(t, v) / seqLen);
      }
      dLogits.set(t, targets[t], dLogits.get(t, targets[t]) - 1 / seqLen);
    }
    
    return { loss: totalLoss, dLogits };
  }

  /**
   * Backward pass
   */
  backward(dLogits) {
    // Output projection backward
    this.dOutputW = matmul(transpose(this._finalHidden), dLogits);
    this.dOutputB = sumRows(dLogits);
    
    let dH = matmul(dLogits, transpose(this.outputW));
    
    // Backward through decoder blocks (reverse order)
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      dH = this.blocks[i].backward(dH);
    }
    
    // Embedding gradient
    this.dEmbedding = new Matrix(this.vocabSize, this.dModel);
    for (let t = 0; t < this._tokens.length; t++) {
      const tokId = this._tokens[t];
      for (let d = 0; d < this.dModel; d++) {
        this.dEmbedding.set(tokId, d, this.dEmbedding.get(tokId, d) + dH.get(t, d));
      }
    }
  }

  /**
   * Update all parameters
   */
  update(lr) {
    updateWeight(this.embedding, this.dEmbedding, lr);
    for (const block of this.blocks) block.update(lr);
    updateWeight(this.outputW, this.dOutputW, lr);
    updateWeight(this.outputB, this.dOutputB, lr);
  }

  /**
   * Update all parameters using AdamW optimizer
   */
  updateAdamW(optimizer) {
    optimizer.update('embedding', this.embedding, this.dEmbedding);
    optimizer.update('outputW', this.outputW, this.dOutputW);
    optimizer.update('outputB', this.outputB, this.dOutputB);
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      optimizer.update(`b${i}.attn.Wq`, b.attn.Wq, b.attn.dWq);
      optimizer.update(`b${i}.attn.Wk`, b.attn.Wk, b.attn.dWk);
      optimizer.update(`b${i}.attn.Wv`, b.attn.Wv, b.attn.dWv);
      optimizer.update(`b${i}.attn.Wo`, b.attn.Wo, b.attn.dWo);
      optimizer.update(`b${i}.ffn.W1`, b.ffn.W1, b.ffn.dW1);
      optimizer.update(`b${i}.ffn.b1`, b.ffn.b1, b.ffn.db1);
      optimizer.update(`b${i}.ffn.W2`, b.ffn.W2, b.ffn.dW2);
      optimizer.update(`b${i}.ffn.b2`, b.ffn.b2, b.ffn.db2);
    }
  }

  /**
   * Train on a single sequence: tokens[0..n-1] predict tokens[1..n]
   */
  trainStep(tokens, lr = 0.001, clipNorm = 1.0, optimizer = null) {
    const input = tokens.slice(0, -1);
    const target = tokens.slice(1);
    
    const logits = this.forward(input);
    const { loss, dLogits } = this.loss(logits, target);
    this.backward(dLogits);
    
    // Gradient clipping by global norm
    if (clipNorm > 0) this._clipGradients(clipNorm);
    
    if (optimizer) {
      this.updateAdamW(optimizer);
    } else {
      this.update(lr);
    }
    
    return loss;
  }

  /**
   * Forward + backward without weight update (for gradient accumulation).
   * Gradients are ADDED to existing gradients, not replaced.
   */
  trainStepAccumulate(tokens) {
    const input = tokens.slice(0, -1);
    const target = tokens.slice(1);
    
    const logits = this.forward(input);
    const { loss, dLogits } = this.loss(logits, target);
    
    // Save current gradients
    const savedGrads = this._saveGradients();
    
    this.backward(dLogits);
    
    // Add saved gradients back (accumulate)
    this._addGradients(savedGrads);
    
    return loss;
  }

  /** Save current gradient state */
  _saveGradients() {
    const saved = {
      dEmbedding: this.dEmbedding ? new Float64Array(this.dEmbedding.data) : null,
      dOutputW: this.dOutputW ? new Float64Array(this.dOutputW.data) : null,
      dOutputB: this.dOutputB ? new Float64Array(this.dOutputB.data) : null,
      blocks: this.blocks.map(b => ({
        dWq: b.attn.dWq ? new Float64Array(b.attn.dWq.data) : null,
        dWk: b.attn.dWk ? new Float64Array(b.attn.dWk.data) : null,
        dWv: b.attn.dWv ? new Float64Array(b.attn.dWv.data) : null,
        dWo: b.attn.dWo ? new Float64Array(b.attn.dWo.data) : null,
        dW1: b.ffn.dW1 ? new Float64Array(b.ffn.dW1.data) : null,
        db1: b.ffn.db1 ? new Float64Array(b.ffn.db1.data) : null,
        dW2: b.ffn.dW2 ? new Float64Array(b.ffn.dW2.data) : null,
        db2: b.ffn.db2 ? new Float64Array(b.ffn.db2.data) : null,
      })),
    };
    return saved;
  }

  /** Add saved gradients to current gradients (accumulation) */
  _addGradients(saved) {
    if (saved.dEmbedding && this.dEmbedding) {
      for (let i = 0; i < this.dEmbedding.data.length; i++) this.dEmbedding.data[i] += saved.dEmbedding[i];
    }
    if (saved.dOutputW && this.dOutputW) {
      for (let i = 0; i < this.dOutputW.data.length; i++) this.dOutputW.data[i] += saved.dOutputW[i];
    }
    if (saved.dOutputB && this.dOutputB) {
      for (let i = 0; i < this.dOutputB.data.length; i++) this.dOutputB.data[i] += saved.dOutputB[i];
    }
    for (let bi = 0; bi < this.blocks.length; bi++) {
      const b = this.blocks[bi];
      const s = saved.blocks[bi];
      const pairs = [
        [b.attn.dWq, s.dWq], [b.attn.dWk, s.dWk], [b.attn.dWv, s.dWv], [b.attn.dWo, s.dWo],
        [b.ffn.dW1, s.dW1], [b.ffn.db1, s.db1], [b.ffn.dW2, s.dW2], [b.ffn.db2, s.db2],
      ];
      for (const [curr, prev] of pairs) {
        if (curr && prev) {
          for (let i = 0; i < curr.data.length; i++) curr.data[i] += prev[i];
        }
      }
    }
  }

  /** Scale all current gradients by a factor (for averaging accumulated gradients) */
  _scaleGradients(factor) {
    const scale = (mat) => { if (mat) for (let i = 0; i < mat.data.length; i++) mat.data[i] *= factor; };
    scale(this.dEmbedding);
    scale(this.dOutputW);
    scale(this.dOutputB);
    for (const b of this.blocks) {
      scale(b.attn.dWq); scale(b.attn.dWk); scale(b.attn.dWv); scale(b.attn.dWo);
      scale(b.ffn.dW1); scale(b.ffn.db1); scale(b.ffn.dW2); scale(b.ffn.db2);
    }
  }

  /**
   * Clip all gradients by global norm to prevent exploding gradients
   */
  _clipGradients(maxNorm) {
    // Collect all gradient matrices
    const grads = [this.dEmbedding, this.dOutputW, this.dOutputB];
    for (const block of this.blocks) {
      grads.push(block.attn.dWq, block.attn.dWk, block.attn.dWv, block.attn.dWo);
      grads.push(block.ffn.dW1, block.ffn.db1, block.ffn.dW2, block.ffn.db2);
    }
    
    // Compute global norm
    let totalNormSq = 0;
    for (const g of grads) {
      for (let i = 0; i < g.rows; i++) {
        for (let j = 0; j < g.cols; j++) {
          totalNormSq += g.get(i, j) ** 2;
        }
      }
    }
    const totalNorm = Math.sqrt(totalNormSq);
    
    // Scale gradients if norm exceeds max
    if (totalNorm > maxNorm) {
      const scale = maxNorm / totalNorm;
      for (const g of grads) {
        for (let i = 0; i < g.rows; i++) {
          for (let j = 0; j < g.cols; j++) {
            g.set(i, j, g.get(i, j) * scale);
          }
        }
      }
    }
  }

  /**
   * Train on text data for multiple epochs
   * @param {number[]} allTokens - Full tokenized text
   * @param {Object} opts - Training options
   * @returns {{ losses: number[], finalLoss: number }}
   */
  train(allTokens, { epochs = 1000, windowSize = 16, lr = 0.001, clipNorm = 1.0, logEvery = 100, optimizer = null, lrSchedule = null, gradAccumSteps = 1 } = {}) {
    // Create AdamW optimizer if requested by string
    let opt = optimizer;
    if (optimizer === 'adamw') {
      opt = new AdamW({ lr, weightDecay: 0.01 });
    } else if (optimizer === 'adamw-warmup') {
      opt = new AdamW({ lr, weightDecay: 0.01 });
      opt._warmupSteps = Math.min(100, Math.floor(epochs * 0.1));
    }
    
    // Create LR scheduler if requested
    let scheduler = lrSchedule;
    if (lrSchedule === 'cosine') {
      const cosine = new CosineAnnealingLR(lr, epochs, lr * 0.01);
      let step = 0;
      scheduler = { step() { return cosine.getLR(step++); } };
    } else if (lrSchedule === 'cosine-warmup') {
      const warmupSteps = Math.min(100, Math.floor(epochs * 0.1));
      const cosine = new CosineAnnealingLR(lr, epochs - warmupSteps, lr * 0.01);
      let currentStep = 0;
      scheduler = { 
        step() {
          const i = currentStep++;
          if (i < warmupSteps) return lr * (i + 1) / warmupSteps;
          return cosine.getLR(i - warmupSteps);
        }
      };
    }
    
    const losses = [];
    for (let i = 0; i < epochs; i++) {
      const start = Math.floor(Math.random() * (allTokens.length - windowSize));
      const window = allTokens.slice(start, start + windowSize);
      
      // Learning rate scheduling
      let currentLr = lr;
      if (scheduler && scheduler.step) {
        currentLr = scheduler.step(i);
      } else if (opt && opt._warmupSteps && i < opt._warmupSteps) {
        currentLr = lr * (i + 1) / opt._warmupSteps;
      }
      
      if (opt) opt.lr = currentLr;
      
      if (gradAccumSteps > 1) {
        // Gradient accumulation: forward+backward N times, then update once
        let accumLoss = 0;
        for (let ga = 0; ga < gradAccumSteps; ga++) {
          const gaStart = Math.floor(Math.random() * (allTokens.length - windowSize));
          const gaWindow = allTokens.slice(gaStart, gaStart + windowSize);
          if (ga === 0) {
            // First step: normal forward+backward (sets gradients)
            const input = gaWindow.slice(0, -1);
            const target = gaWindow.slice(1);
            const logits = this.forward(input);
            const { loss: l, dLogits } = this.loss(logits, target);
            this.backward(dLogits);
            accumLoss += l;
          } else {
            // Subsequent steps: accumulate gradients
            accumLoss += this.trainStepAccumulate(gaWindow);
          }
        }
        // Average gradients
        this._scaleGradients(1.0 / gradAccumSteps);
        if (clipNorm > 0) this._clipGradients(clipNorm);
        if (opt) {
          this.updateAdamW(opt);
        } else {
          this.update(currentLr);
        }
        losses.push(accumLoss / gradAccumSteps);
      } else {
        const loss = this.trainStep(window, currentLr, clipNorm, opt);
        losses.push(loss);
      }
      if (logEvery > 0 && i % logEvery === 0) {
        const avg = losses.slice(-logEvery).reduce((a, b) => a + b) / Math.min(logEvery, losses.length);
        console.log(`Step ${i}: loss = ${avg.toFixed(4)} lr = ${currentLr.toFixed(6)}`);
      }
    }
    const finalLoss = losses.slice(-100).reduce((a, b) => a + b) / Math.min(100, losses.length);
    return { losses, finalLoss };
  }

  /**
   * Generate text autoregressively
   * @param {number[]} prompt - Starting token IDs
   * @param {number} maxTokens - How many tokens to generate
   * @param {number} temperature - Sampling temperature (1.0 = normal, <1 = sharper, >1 = more random)
   * @param {Object} opts - Additional options: topK, topP (nucleus sampling)
   * @returns {number[]} Generated token IDs
   */
  generate(prompt, maxTokens = 50, temperature = 0.8, { topK = 0, topP = 0 } = {}) {
    const tokens = [...prompt];
    
    for (let i = 0; i < maxTokens; i++) {
      // Use last maxLen tokens as context
      const context = tokens.slice(-this.maxLen);
      const logits = this.forward(context);
      
      // Get logits for last position
      const lastRow = logits.rows - 1;
      let lastLogits = [];
      for (let v = 0; v < this.vocabSize; v++) {
        lastLogits.push(logits.get(lastRow, v) / temperature);
      }
      
      // Top-K filtering: keep only K highest logits
      if (topK > 0 && topK < this.vocabSize) {
        const indexed = lastLogits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        const threshold = indexed[topK - 1][0];
        lastLogits = lastLogits.map(v => v >= threshold ? v : -1e9);
      }
      
      // Top-P (nucleus) filtering: keep smallest set whose cumulative prob >= topP
      if (topP > 0 && topP < 1) {
        const probs = softmax(lastLogits);
        const indexed = probs.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        let cumSum = 0;
        const keep = new Set();
        for (const [p, idx] of indexed) {
          cumSum += p;
          keep.add(idx);
          if (cumSum >= topP) break;
        }
        lastLogits = lastLogits.map((v, i) => keep.has(i) ? v : -1e9);
      }
      
      // Sample from softmax distribution
      const probs = softmax(lastLogits);
      const nextToken = sample(probs);
      tokens.push(nextToken);
    }
    
    return tokens;
  }

  paramCount() {
    let count = this.vocabSize * this.dModel; // embedding
    count += this.dModel * this.vocabSize + this.vocabSize; // output proj
    for (const block of this.blocks) {
      const attn = block.attn;
      count += 4 * this.dModel * this.dModel; // Wq, Wk, Wv, Wo
      const ffn = block.ffn;
      count += ffn.W1.rows * ffn.W1.cols + ffn.b1.cols;
      count += ffn.W2.rows * ffn.W2.cols + ffn.b2.cols;
    }
    return count;
  }
}

// --- Matrix utilities ---

function matmul(a, b) {
  if (a.cols !== b.rows) throw new Error(`matmul: ${a.rows}x${a.cols} * ${b.rows}x${b.cols}`);
  const result = new Matrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += a.get(i, k) * b.get(k, j);
      }
      result.set(i, j, sum);
    }
  }
  return result;
}

function transpose(m) {
  const result = new Matrix(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      result.set(j, i, m.get(i, j));
    }
  }
  return result;
}

function add(a, b) {
  const result = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      result.set(i, j, a.get(i, j) + b.get(i, j));
    }
  }
  return result;
}

function addInPlace(a, b) {
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      a.set(i, j, a.get(i, j) + b.get(i, j));
    }
  }
}

function addBias(m, bias) {
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      m.set(i, j, m.get(i, j) + bias.get(0, j));
    }
  }
}

function relu(m) {
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) {
      if (m.get(i, j) < 0) m.set(i, j, 0);
    }
  }
}

function softmaxRows(m) {
  const result = new Matrix(m.rows, m.cols);
  for (let i = 0; i < m.rows; i++) {
    let max = -Infinity;
    for (let j = 0; j < m.cols; j++) max = Math.max(max, m.get(i, j));
    let sum = 0;
    for (let j = 0; j < m.cols; j++) {
      const val = Math.exp(m.get(i, j) - max);
      result.set(i, j, val);
      sum += val;
    }
    for (let j = 0; j < m.cols; j++) result.set(i, j, result.get(i, j) / sum);
  }
  return result;
}

function softmaxBackward(probs, dOutput) {
  const result = new Matrix(probs.rows, probs.cols);
  for (let i = 0; i < probs.rows; i++) {
    for (let j = 0; j < probs.cols; j++) {
      let sum = 0;
      for (let k = 0; k < probs.cols; k++) {
        if (j === k) {
          sum += probs.get(i, j) * (1 - probs.get(i, j)) * dOutput.get(i, k);
        } else {
          sum += -probs.get(i, j) * probs.get(i, k) * dOutput.get(i, k);
        }
      }
      result.set(i, j, sum);
    }
  }
  return result;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function sample(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function sumRows(m) {
  const result = new Matrix(1, m.cols);
  for (let j = 0; j < m.cols; j++) {
    let sum = 0;
    for (let i = 0; i < m.rows; i++) sum += m.get(i, j);
    result.set(0, j, sum);
  }
  return result;
}

function updateWeight(w, dw, lr) {
  for (let i = 0; i < w.rows; i++) {
    for (let j = 0; j < w.cols; j++) {
      w.set(i, j, w.get(i, j) - lr * dw.get(i, j));
    }
  }
}
