// mamba-ssm.js — Selective State Space Model (Gu & Dao, 2023)
// "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//
// Key insight: make SSM parameters (B, C, Δ) input-dependent.
// This allows the model to selectively focus on or ignore inputs.
//
// Continuous → Discrete (zero-order hold):
//   Ā = exp(Δ * A)
//   B̄ = (Δ * A)^{-1} * (Ā - I) * Δ * B  ≈  Δ * B  (simplified)
//
// Recurrence: h_t = Ā * h_{t-1} + B̄ * x_t
//             y_t = C * h_t

import { Matrix } from './matrix.js';

export class SelectiveSSM {
  /**
   * @param {number} dModel - Input/output dimension
   * @param {number} dState - Latent state dimension (N in the paper)
   * @param {number} dInner - Inner dimension for expansion
   */
  constructor(dModel, dState = 16, dInner = null) {
    this.dModel = dModel;
    this.dState = dState;
    this.dInner = dInner || dModel * 2; // Typical expansion factor
    
    // Input projection: x → (z, x_inner) via linear
    this.Wz = Matrix.random(dModel, this.dInner).map(v => v * Math.sqrt(2.0 / dModel));
    this.Wx = Matrix.random(dModel, this.dInner).map(v => v * Math.sqrt(2.0 / dModel));
    
    // SSM parameters (input-dependent)
    // B and C are projected from input
    this.WB = Matrix.random(this.dInner, dState).map(v => v * Math.sqrt(2.0 / this.dInner));
    this.WC = Matrix.random(this.dInner, dState).map(v => v * Math.sqrt(2.0 / this.dInner));
    
    // Δ (discretization step) is also input-dependent
    this.WDelta = Matrix.random(this.dInner, 1).map(v => v * 0.01);
    this.deltaMin = 0.001;
    this.deltaMax = 0.1;
    
    // A: structured state matrix (diagonal, initialized to -1, -2, ..., -N)
    this.A = new Float64Array(dState);
    for (let i = 0; i < dState; i++) this.A[i] = -(i + 1);
    
    // D: skip connection (residual)
    this.D = new Float64Array(this.dInner).fill(1);
    
    // Output projection
    this.Wout = Matrix.random(this.dInner, dModel).map(v => v * Math.sqrt(2.0 / this.dInner));
  }

  /**
   * Selective scan: process sequence with input-dependent parameters.
   * @param {Matrix} x - Input (seqLen × dModel)
   * @returns {Matrix} Output (seqLen × dModel)
   */
  forward(x) {
    const seqLen = x.rows;
    const dInner = this.dInner;
    const dState = this.dState;
    
    // Project input
    const z = new Matrix(seqLen, dInner); // Gate
    const xInner = new Matrix(seqLen, dInner); // SSM input
    
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dInner; d++) {
        let sz = 0, sx = 0;
        for (let i = 0; i < this.dModel; i++) {
          sz += x.get(t, i) * this.Wz.get(i, d);
          sx += x.get(t, i) * this.Wx.get(i, d);
        }
        z.set(t, d, silu(sz)); // SiLU gating
        xInner.set(t, d, sx);
      }
    }
    
    // Compute input-dependent B, C, Δ
    const B = new Array(seqLen);
    const C = new Array(seqLen);
    const delta = new Array(seqLen);
    
    for (let t = 0; t < seqLen; t++) {
      B[t] = new Float64Array(dState);
      C[t] = new Float64Array(dState);
      for (let n = 0; n < dState; n++) {
        let sb = 0, sc = 0;
        for (let d = 0; d < dInner; d++) {
          sb += xInner.get(t, d) * this.WB.get(d, n);
          sc += xInner.get(t, d) * this.WC.get(d, n);
        }
        B[t][n] = sb;
        C[t][n] = sc;
      }
      
      // Δ: softplus to ensure positive, then clamp
      let rawDelta = 0;
      for (let d = 0; d < dInner; d++) rawDelta += xInner.get(t, d) * this.WDelta.get(d, 0);
      delta[t] = Math.max(this.deltaMin, Math.min(this.deltaMax, softplus(rawDelta)));
    }
    
    // Selective scan (per-dimension recurrence)
    const y = new Matrix(seqLen, dInner);
    
    // For each inner dimension, run the SSM recurrence
    for (let d = 0; d < dInner; d++) {
      const h = new Float64Array(dState); // State
      
      for (let t = 0; t < seqLen; t++) {
        const dt = delta[t];
        
        // Discretize A → Ā = exp(Δ * A)  (diagonal, so element-wise)
        // Update state: h = Ā * h + B̄ * x
        for (let n = 0; n < dState; n++) {
          const aBar = Math.exp(dt * this.A[n]);
          const bBar = dt * B[t][n]; // Simplified discretization
          h[n] = aBar * h[n] + bBar * xInner.get(t, d);
        }
        
        // Output: y = C * h + D * x
        let yt = this.D[d] * xInner.get(t, d);
        for (let n = 0; n < dState; n++) {
          yt += C[t][n] * h[n];
        }
        y.set(t, d, yt);
      }
    }
    
    // Apply gate: y = y ⊙ z
    for (let t = 0; t < seqLen; t++) {
      for (let d = 0; d < dInner; d++) {
        y.set(t, d, y.get(t, d) * z.get(t, d));
      }
    }
    
    // Output projection
    const output = new Matrix(seqLen, this.dModel);
    for (let t = 0; t < seqLen; t++) {
      for (let j = 0; j < this.dModel; j++) {
        let sum = 0;
        for (let d = 0; d < dInner; d++) sum += y.get(t, d) * this.Wout.get(d, j);
        output.set(t, j, sum);
      }
    }
    
    return output;
  }

  /**
   * Single-step inference with state.
   */
  step(xt, state) {
    const dInner = this.dInner;
    const dState = this.dState;
    
    // Project
    const zVec = new Float64Array(dInner);
    const xVec = new Float64Array(dInner);
    for (let d = 0; d < dInner; d++) {
      let sz = 0, sx = 0;
      for (let i = 0; i < this.dModel; i++) {
        sz += xt[i] * this.Wz.get(i, d);
        sx += xt[i] * this.Wx.get(i, d);
      }
      zVec[d] = silu(sz);
      xVec[d] = sx;
    }
    
    // B, C, Δ
    const bVec = new Float64Array(dState);
    const cVec = new Float64Array(dState);
    for (let n = 0; n < dState; n++) {
      let sb = 0, sc = 0;
      for (let d = 0; d < dInner; d++) {
        sb += xVec[d] * this.WB.get(d, n);
        sc += xVec[d] * this.WC.get(d, n);
      }
      bVec[n] = sb;
      cVec[n] = sc;
    }
    let rawDelta = 0;
    for (let d = 0; d < dInner; d++) rawDelta += xVec[d] * this.WDelta.get(d, 0);
    const dt = Math.max(this.deltaMin, Math.min(this.deltaMax, softplus(rawDelta)));
    
    // State update per dimension
    const newState = state.map(h => new Float64Array(h));
    const yVec = new Float64Array(dInner);
    
    for (let d = 0; d < dInner; d++) {
      for (let n = 0; n < dState; n++) {
        const aBar = Math.exp(dt * this.A[n]);
        const bBar = dt * bVec[n];
        newState[d][n] = aBar * newState[d][n] + bBar * xVec[d];
      }
      
      let yt = this.D[d] * xVec[d];
      for (let n = 0; n < dState; n++) yt += cVec[n] * newState[d][n];
      yVec[d] = yt * zVec[d];
    }
    
    // Output
    const output = new Float64Array(this.dModel);
    for (let j = 0; j < this.dModel; j++) {
      for (let d = 0; d < dInner; d++) output[j] += yVec[d] * this.Wout.get(d, j);
    }
    
    return { output, state: newState };
  }

  initState() {
    return Array.from({ length: this.dInner }, () => new Float64Array(this.dState));
  }
}

function silu(x) { return x / (1 + Math.exp(-x)); }
function softplus(x) { return Math.log(1 + Math.exp(x)); }
