// swiglu.js — SwiGLU Activation (Shazeer 2020)
// Used in LLaMA, Mistral, PaLM. More expressive than ReLU/GELU.
// SwiGLU(x, W1, Wgate) = (x @ W1) ⊙ swish(x @ Wgate)
// where swish(x) = x * sigmoid(x) = x * σ(x)

import { Matrix } from './matrix.js';

/**
 * Swish activation: x * sigmoid(x)
 */
export function swish(x) {
  return x / (1 + Math.exp(-x));
}

/**
 * Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 */
export function swishDerivative(x) {
  const sig = 1 / (1 + Math.exp(-x));
  return sig + x * sig * (1 - sig);
}

/**
 * SwiGLU FFN Layer.
 * Replaces the standard FFN (linear → ReLU → linear) with:
 *   output = (x @ W1) ⊙ swish(x @ Wgate) → @ W2
 */
export class SwiGLU {
  /**
   * @param {number} inputDim - Input dimension
   * @param {number} hiddenDim - Hidden dimension
   */
  constructor(inputDim, hiddenDim) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    
    // W1: linear projection (input → hidden)
    this.W1 = Matrix.random(inputDim, hiddenDim).map(v => v * Math.sqrt(2.0 / inputDim));
    this.b1 = new Matrix(1, hiddenDim);
    
    // Wgate: gate projection (input → hidden)
    this.Wgate = Matrix.random(inputDim, hiddenDim).map(v => v * Math.sqrt(2.0 / inputDim));
    this.bgate = new Matrix(1, hiddenDim);
    
    // W2: output projection (hidden → input)
    this.W2 = Matrix.random(hiddenDim, inputDim).map(v => v * Math.sqrt(2.0 / hiddenDim));
    this.b2 = new Matrix(1, inputDim);
    
    // Gradients
    this.dW1 = new Matrix(inputDim, hiddenDim);
    this.db1 = new Matrix(1, hiddenDim);
    this.dWgate = new Matrix(inputDim, hiddenDim);
    this.dbgate = new Matrix(1, hiddenDim);
    this.dW2 = new Matrix(hiddenDim, inputDim);
    this.db2 = new Matrix(1, inputDim);
    
    // Saved for backward
    this._input = null;
    this._linear = null;  // x @ W1 + b1
    this._gate = null;    // swish(x @ Wgate + bgate)
    this._gateRaw = null; // x @ Wgate + bgate (pre-swish)
  }

  forward(x) {
    this._input = x;
    const batch = x.rows;
    
    // Linear: x @ W1 + b1
    this._linear = new Matrix(batch, this.hiddenDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        let sum = this.b1.get(0, j);
        for (let k = 0; k < this.inputDim; k++) {
          sum += x.get(i, k) * this.W1.get(k, j);
        }
        this._linear.set(i, j, sum);
      }
    }
    
    // Gate: swish(x @ Wgate + bgate)
    this._gateRaw = new Matrix(batch, this.hiddenDim);
    this._gate = new Matrix(batch, this.hiddenDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        let sum = this.bgate.get(0, j);
        for (let k = 0; k < this.inputDim; k++) {
          sum += x.get(i, k) * this.Wgate.get(k, j);
        }
        this._gateRaw.set(i, j, sum);
        this._gate.set(i, j, swish(sum));
      }
    }
    
    // SwiGLU: linear ⊙ gate → @ W2 + b2
    const hidden = new Matrix(batch, this.hiddenDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        hidden.set(i, j, this._linear.get(i, j) * this._gate.get(i, j));
      }
    }
    
    const output = new Matrix(batch, this.inputDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.inputDim; j++) {
        let sum = this.b2.get(0, j);
        for (let k = 0; k < this.hiddenDim; k++) {
          sum += hidden.get(i, k) * this.W2.get(k, j);
        }
        output.set(i, j, sum);
      }
    }
    
    this._hidden = hidden;
    return output;
  }

  backward(dOutput) {
    const batch = dOutput.rows;
    
    // dW2 = hidden^T @ dOutput
    this.dW2 = new Matrix(this.hiddenDim, this.inputDim);
    for (let k = 0; k < this.hiddenDim; k++) {
      for (let j = 0; j < this.inputDim; j++) {
        let sum = 0;
        for (let i = 0; i < batch; i++) sum += this._hidden.get(i, k) * dOutput.get(i, j);
        this.dW2.set(k, j, sum);
      }
    }
    this.db2 = new Matrix(1, this.inputDim);
    for (let j = 0; j < this.inputDim; j++) {
      let sum = 0;
      for (let i = 0; i < batch; i++) sum += dOutput.get(i, j);
      this.db2.set(0, j, sum);
    }
    
    // dHidden = dOutput @ W2^T
    const dHidden = new Matrix(batch, this.hiddenDim);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < this.hiddenDim; k++) {
        let sum = 0;
        for (let j = 0; j < this.inputDim; j++) sum += dOutput.get(i, j) * this.W2.get(k, j);
        dHidden.set(i, k, sum);
      }
    }
    
    // SwiGLU backward: dLinear = dHidden ⊙ gate, dGate = dHidden ⊙ linear
    const dLinear = new Matrix(batch, this.hiddenDim);
    const dGateRaw = new Matrix(batch, this.hiddenDim);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        dLinear.set(i, j, dHidden.get(i, j) * this._gate.get(i, j));
        const dGate = dHidden.get(i, j) * this._linear.get(i, j);
        dGateRaw.set(i, j, dGate * swishDerivative(this._gateRaw.get(i, j)));
      }
    }
    
    // dW1 = input^T @ dLinear, dWgate = input^T @ dGateRaw
    this.dW1 = new Matrix(this.inputDim, this.hiddenDim);
    this.dWgate = new Matrix(this.inputDim, this.hiddenDim);
    for (let k = 0; k < this.inputDim; k++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        let sum1 = 0, sumG = 0;
        for (let i = 0; i < batch; i++) {
          sum1 += this._input.get(i, k) * dLinear.get(i, j);
          sumG += this._input.get(i, k) * dGateRaw.get(i, j);
        }
        this.dW1.set(k, j, sum1);
        this.dWgate.set(k, j, sumG);
      }
    }
    
    // dInput = dLinear @ W1^T + dGateRaw @ Wgate^T
    const dInput = new Matrix(batch, this.inputDim);
    for (let i = 0; i < batch; i++) {
      for (let k = 0; k < this.inputDim; k++) {
        let sum = 0;
        for (let j = 0; j < this.hiddenDim; j++) {
          sum += dLinear.get(i, j) * this.W1.get(k, j);
          sum += dGateRaw.get(i, j) * this.Wgate.get(k, j);
        }
        dInput.set(i, k, sum);
      }
    }
    
    return dInput;
  }

  update(lr) {
    const params = [
      [this.W1, this.dW1], [this.b1, this.db1],
      [this.Wgate, this.dWgate], [this.bgate, this.dbgate],
      [this.W2, this.dW2], [this.b2, this.db2],
    ];
    for (const [w, dw] of params) {
      for (let i = 0; i < w.data.length; i++) w.data[i] -= lr * dw.data[i];
    }
  }

  paramCount() {
    return (this.inputDim * this.hiddenDim + this.hiddenDim) * 2 // W1+b1 + Wgate+bgate
         + this.hiddenDim * this.inputDim + this.inputDim; // W2+b2
  }
}
