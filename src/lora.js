// lora.js — Low-Rank Adaptation (Hu et al., 2021)
// Efficient fine-tuning by adding low-rank decomposition to weight matrices.
//
// Instead of updating W (d×d), add ΔW = B*A where:
//   A is (r×d) and B is (d×r), with r << d
//   A initialized randomly, B initialized to zero
//   Only A and B are trained; original W is frozen
//
// Parameter savings: d² → 2*d*r (typically r=4-16)

import { Matrix } from './matrix.js';

export class LoRALayer {
  /**
   * @param {Matrix} baseWeight - Original weight matrix (frozen)
   * @param {number} rank - Low-rank dimension (typically 4-16)
   * @param {number} alpha - Scaling factor (default: rank)
   */
  constructor(baseWeight, rank = 4, alpha = null) {
    this.baseWeight = baseWeight; // Frozen original weight
    this.rank = rank;
    this.alpha = alpha || rank;
    this.scaling = this.alpha / this.rank;
    
    const [rows, cols] = [baseWeight.rows, baseWeight.cols];
    
    // A: (rank × cols), initialized with small random values
    this.A = new Matrix(rank, cols);
    for (let i = 0; i < rank * cols; i++) {
      this.A.data[i] = (Math.random() * 2 - 1) * Math.sqrt(2.0 / cols);
    }
    
    // B: (rows × rank), initialized to zero
    this.B = new Matrix(rows, rank);
    // B starts at zero → initial ΔW = 0 → model starts identical to base
    
    // Gradients
    this.dA = new Matrix(rank, cols);
    this.dB = new Matrix(rows, rank);
    
    // Saved for backward
    this._input = null;
    this._hidden = null; // A @ input
  }

  /**
   * Forward: output = input @ (W + scaling * B @ A)^T
   * Equivalent to: output = input @ W^T + scaling * input @ A^T @ B^T
   * @param {Matrix} input - Input (batchSize × cols)
   * @returns {Matrix} Output (batchSize × rows)
   */
  forward(input) {
    this._input = input;
    const batch = input.rows;
    const outDim = this.baseWeight.rows;
    const inDim = this.baseWeight.cols;
    
    // Base: input @ W^T
    const baseOut = new Matrix(batch, outDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < outDim; j++) {
        let sum = 0;
        for (let k = 0; k < inDim; k++) {
          sum += input.get(i, k) * this.baseWeight.get(j, k);
        }
        baseOut.set(i, j, sum);
      }
    }
    
    // LoRA: input @ A^T @ B^T * scaling
    // Step 1: hidden = input @ A^T → (batch × rank)
    this._hidden = new Matrix(batch, this.rank);
    for (let i = 0; i < batch; i++) {
      for (let r = 0; r < this.rank; r++) {
        let sum = 0;
        for (let k = 0; k < inDim; k++) {
          sum += input.get(i, k) * this.A.get(r, k);
        }
        this._hidden.set(i, r, sum);
      }
    }
    
    // Step 2: loraOut = hidden @ B^T * scaling → (batch × outDim)
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < outDim; j++) {
        let sum = 0;
        for (let r = 0; r < this.rank; r++) {
          sum += this._hidden.get(i, r) * this.B.get(j, r);
        }
        baseOut.set(i, j, baseOut.get(i, j) + sum * this.scaling);
      }
    }
    
    return baseOut;
  }

  /**
   * Backward: compute gradients for A and B (base weight is frozen).
   * @param {Matrix} dOutput - Gradient (batch × outDim)
   * @returns {Matrix} Gradient w.r.t. input (batch × inDim)
   */
  backward(dOutput) {
    const batch = dOutput.rows;
    const outDim = this.baseWeight.rows;
    const inDim = this.baseWeight.cols;
    
    // dB = dOutput^T @ hidden * scaling → (outDim × rank)
    this.dB = new Matrix(outDim, this.rank);
    for (let j = 0; j < outDim; j++) {
      for (let r = 0; r < this.rank; r++) {
        let sum = 0;
        for (let i = 0; i < batch; i++) {
          sum += dOutput.get(i, j) * this._hidden.get(i, r);
        }
        this.dB.set(j, r, sum * this.scaling);
      }
    }
    
    // dHidden = dOutput @ B * scaling → (batch × rank)
    const dHidden = new Matrix(batch, this.rank);
    for (let i = 0; i < batch; i++) {
      for (let r = 0; r < this.rank; r++) {
        let sum = 0;
        for (let j = 0; j < outDim; j++) {
          sum += dOutput.get(i, j) * this.B.get(j, r);
        }
        dHidden.set(i, r, sum * this.scaling);
      }
    }
    
    // dA = dHidden^T @ input → (rank × inDim)
    this.dA = new Matrix(this.rank, inDim);
    for (let r = 0; r < this.rank; r++) {
      for (let k = 0; k < inDim; k++) {
        let sum = 0;
        for (let i = 0; i < batch; i++) {
          sum += dHidden.get(i, r) * this._input.get(i, k);
        }
        this.dA.set(r, k, sum);
      }
    }
    
    // dInput = dOutput @ W + dHidden @ A → (batch × inDim)
    const dInput = new Matrix(batch, inDim);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < inDim; k++) {
        let sum = 0;
        // Base gradient
        for (let j = 0; j < outDim; j++) {
          sum += dOutput.get(i, j) * this.baseWeight.get(j, k);
        }
        // LoRA gradient
        for (let r = 0; r < this.rank; r++) {
          sum += dHidden.get(i, r) * this.A.get(r, k);
        }
        dInput.set(i, k, sum);
      }
    }
    
    return dInput;
  }

  update(lr) {
    // Only update A and B (base weight is frozen)
    for (let i = 0; i < this.A.data.length; i++) {
      this.A.data[i] -= lr * this.dA.data[i];
    }
    for (let i = 0; i < this.B.data.length; i++) {
      this.B.data[i] -= lr * this.dB.data[i];
    }
  }

  /**
   * Merge LoRA weights into base weight (for inference).
   * Returns W + scaling * B @ A
   */
  merge() {
    const merged = new Matrix(this.baseWeight.rows, this.baseWeight.cols);
    // Copy base weight
    for (let i = 0; i < merged.data.length; i++) {
      merged.data[i] = this.baseWeight.data[i];
    }
    // Add LoRA: B @ A * scaling
    for (let i = 0; i < this.baseWeight.rows; i++) {
      for (let j = 0; j < this.baseWeight.cols; j++) {
        let sum = 0;
        for (let r = 0; r < this.rank; r++) {
          sum += this.B.get(i, r) * this.A.get(r, j);
        }
        merged.set(i, j, merged.get(i, j) + sum * this.scaling);
      }
    }
    return merged;
  }

  paramCount() {
    return this.A.data.length + this.B.data.length;
  }

  baseParamCount() {
    return this.baseWeight.data.length;
  }

  savings() {
    const lora = this.paramCount();
    const base = this.baseParamCount();
    return {
      loraParams: lora,
      baseParams: base,
      ratio: (lora / base * 100).toFixed(2) + '%',
      savings: ((1 - lora / base) * 100).toFixed(2) + '%',
    };
  }
}
