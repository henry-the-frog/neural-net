// rwkv.js — RWKV Linear Attention (Peng et al., 2023)
// "Reinventing RNNs for the Transformer Era"
// O(N) time and O(1) memory per token during inference.
// Combines RNN efficiency with transformer quality.
//
// Key formulas:
// r_t = σ(W_r · x_t)         (receptance gate)
// k_t = W_k · x_t             (key)
// v_t = W_v · x_t             (value)
// wkv_t = (a_{t-1} + e^{u+k_t} · v_t) / (b_{t-1} + e^{u+k_t})
// output_t = r_t ⊙ wkv_t
// a_t = e^{-w} · a_{t-1} + e^{k_t} · v_t  (numerator state)
// b_t = e^{-w} · b_{t-1} + e^{k_t}          (denominator state)

import { Matrix } from './matrix.js';

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export class RWKVTimeBlock {
  /**
   * @param {number} dim - Model dimension
   */
  constructor(dim) {
    this.dim = dim;
    
    // Projection weights
    this.Wr = Matrix.random(dim, dim).map(v => v * Math.sqrt(2.0 / dim));
    this.Wk = Matrix.random(dim, dim).map(v => v * Math.sqrt(2.0 / dim));
    this.Wv = Matrix.random(dim, dim).map(v => v * Math.sqrt(2.0 / dim));
    this.Wo = Matrix.random(dim, dim).map(v => v * Math.sqrt(2.0 / dim));
    
    // Time decay: w (per-dim, learned, positive)
    this.w = new Float64Array(dim);
    for (let i = 0; i < dim; i++) this.w[i] = 0.5 + Math.random() * 0.5;
    
    // Bonus: u (per-dim, learned)
    this.u = new Float64Array(dim);
    for (let i = 0; i < dim; i++) this.u[i] = Math.random() * 0.1;
  }

  /**
   * Forward pass: process entire sequence.
   * @param {Matrix} x - Input (seqLen × dim)
   * @returns {Matrix} Output (seqLen × dim)
   */
  forward(x) {
    const seqLen = x.rows;
    const dim = this.dim;
    
    // Project to r, k, v
    const r = new Matrix(seqLen, dim);
    const k = new Matrix(seqLen, dim);
    const v = new Matrix(seqLen, dim);
    
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dim; d++) {
        let sr = 0, sk = 0, sv = 0;
        for (let i = 0; i < dim; i++) {
          const xi = x.get(t, i);
          sr += xi * this.Wr.get(i, d);
          sk += xi * this.Wk.get(i, d);
          sv += xi * this.Wv.get(i, d);
        }
        r.set(t, d, sigmoid(sr)); // Receptance gate
        k.set(t, d, sk);
        v.set(t, d, sv);
      }
    }
    
    // WKV computation (linear recurrence)
    const wkv = new Matrix(seqLen, dim);
    const a = new Float64Array(dim); // Numerator state
    const b = new Float64Array(dim); // Denominator state
    
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dim; d++) {
        const kt = k.get(t, d);
        const vt = v.get(t, d);
        const expUK = Math.exp(Math.min(this.u[d] + kt, 30)); // Clamp for stability
        const expK = Math.exp(Math.min(kt, 30));
        
        // WKV: weighted key-value with current token bonus
        const num = a[d] + expUK * vt;
        const den = b[d] + expUK;
        wkv.set(t, d, den > 1e-10 ? num / den : 0);
        
        // Update state with time decay
        const decay = Math.exp(-this.w[d]);
        a[d] = decay * a[d] + expK * vt;
        b[d] = decay * b[d] + expK;
      }
    }
    
    // Output: receptance gate ⊙ wkv → output projection
    const gated = new Matrix(seqLen, dim);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dim; d++) {
        gated.set(t, d, r.get(t, d) * wkv.get(t, d));
      }
    }
    
    // Output projection
    const output = new Matrix(seqLen, dim);
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let i = 0; i < dim; i++) sum += gated.get(t, i) * this.Wo.get(i, d);
        output.set(t, d, sum);
      }
    }
    
    return output;
  }

  /**
   * Inference mode: process one token at a time with state.
   * @param {Float64Array} xt - Single token input (dim)
   * @param {{ a: Float64Array, b: Float64Array }} state - Recurrent state
   * @returns {{ output: Float64Array, state: { a: Float64Array, b: Float64Array } }}
   */
  step(xt, state) {
    const dim = this.dim;
    const rt = new Float64Array(dim);
    const kt = new Float64Array(dim);
    const vt = new Float64Array(dim);
    
    for (let d = 0; d < dim; d++) {
      let sr = 0, sk = 0, sv = 0;
      for (let i = 0; i < dim; i++) {
        sr += xt[i] * this.Wr.get(i, d);
        sk += xt[i] * this.Wk.get(i, d);
        sv += xt[i] * this.Wv.get(i, d);
      }
      rt[d] = sigmoid(sr);
      kt[d] = sk;
      vt[d] = sv;
    }
    
    const newA = new Float64Array(dim);
    const newB = new Float64Array(dim);
    const gated = new Float64Array(dim);
    
    for (let d = 0; d < dim; d++) {
      const expUK = Math.exp(Math.min(this.u[d] + kt[d], 30));
      const expK = Math.exp(Math.min(kt[d], 30));
      const decay = Math.exp(-this.w[d]);
      
      const num = state.a[d] + expUK * vt[d];
      const den = state.b[d] + expUK;
      const wkvt = den > 1e-10 ? num / den : 0;
      
      gated[d] = rt[d] * wkvt;
      
      newA[d] = decay * state.a[d] + expK * vt[d];
      newB[d] = decay * state.b[d] + expK;
    }
    
    const output = new Float64Array(dim);
    for (let d = 0; d < dim; d++) {
      for (let i = 0; i < dim; i++) output[d] += gated[i] * this.Wo.get(i, d);
    }
    
    return { output, state: { a: newA, b: newB } };
  }

  initState() {
    return {
      a: new Float64Array(this.dim),
      b: new Float64Array(this.dim),
    };
  }
}
