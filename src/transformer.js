// transformer.js — Transformer building blocks: Positional Encoding, Layer Norm, Transformer Encoder

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';
import { MultiHeadAttention } from './attention.js';

/**
 * Sinusoidal Positional Encoding (Vaswani et al. 2017)
 * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 */
export class PositionalEncoding {
  constructor(dModel, maxLen = 512) {
    this.dModel = dModel;
    this.maxLen = maxLen;
    this.outputSize = dModel;
    this.training = true;
    
    // Precompute positional encodings
    this.pe = new Matrix(maxLen, dModel);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        this.pe.set(pos, i, i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
      }
    }
  }
  
  forward(input) {
    // input: [batch, seqLen * dModel]
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const result = new Matrix(batchSize, input.cols);
    
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        for (let d = 0; d < this.dModel; d++) {
          result.set(b, t * this.dModel + d, 
            input.get(b, t * this.dModel + d) + this.pe.get(t, d));
        }
      }
    }
    
    this._input = input;
    return result;
  }
  
  backward(dOutput) { return dOutput; } // PE is additive constant, gradient passes through
  update() {}
  paramCount() { return 0; } // No learnable parameters
}

/**
 * Layer Normalization (Ba et al. 2016)
 * Normalizes across features (not batch) at each position
 */
export class LayerNorm {
  constructor(dModel, epsilon = 1e-5) {
    this.dModel = dModel;
    this.epsilon = epsilon;
    this.outputSize = dModel;
    this.training = true;
    
    // Learnable scale and shift
    this.gamma = new Matrix(1, dModel).map(() => 1.0);
    this.beta = Matrix.zeros(1, dModel);
    
    // Cache
    this._normalized = null;
    this._std = null;
    this.dWeights = null;
    this.dBiases = null;
  }
  
  forward(input) {
    // input: [batch, seqLen * dModel]
    // Normalize per-position (every dModel features)
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const result = new Matrix(batchSize, input.cols);
    this._normalized = new Matrix(batchSize, input.cols);
    this._std = [];
    
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const offset = t * this.dModel;
        // Compute mean and variance
        let mean = 0;
        for (let d = 0; d < this.dModel; d++) mean += input.get(b, offset + d);
        mean /= this.dModel;
        
        let variance = 0;
        for (let d = 0; d < this.dModel; d++) {
          const diff = input.get(b, offset + d) - mean;
          variance += diff * diff;
        }
        variance /= this.dModel;
        
        const std = Math.sqrt(variance + this.epsilon);
        this._std.push(std);
        
        // Normalize, scale, shift
        for (let d = 0; d < this.dModel; d++) {
          const norm = (input.get(b, offset + d) - mean) / std;
          this._normalized.set(b, offset + d, norm);
          result.set(b, offset + d, norm * this.gamma.get(0, d) + this.beta.get(0, d));
        }
      }
    }
    
    this._input = input;
    return result;
  }
  
  backward(dOutput) {
    const batchSize = dOutput.rows;
    const seqLen = Math.floor(dOutput.cols / this.dModel);
    const dInput = new Matrix(batchSize, dOutput.cols);
    
    let dGamma = Matrix.zeros(1, this.dModel);
    let dBeta = Matrix.zeros(1, this.dModel);
    
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        const offset = t * this.dModel;
        const std = this._std[b * seqLen + t];
        
        for (let d = 0; d < this.dModel; d++) {
          const dOut = dOutput.get(b, offset + d);
          const norm = this._normalized.get(b, offset + d);
          dGamma.set(0, d, dGamma.get(0, d) + dOut * norm);
          dBeta.set(0, d, dBeta.get(0, d) + dOut);
          
          // Simplified gradient (ignoring cross-terms for efficiency)
          dInput.set(b, offset + d, dOut * this.gamma.get(0, d) / std);
        }
      }
    }
    
    this.dWeights = dGamma;
    this.dBiases = dBeta;
    this._dGamma = dGamma;
    this._dBeta = dBeta;
    
    return dInput;
  }
  
  update(learningRate) {
    this.gamma = this.gamma.sub(this._dGamma.mul(learningRate));
    this.beta = this.beta.sub(this._dBeta.mul(learningRate));
  }
  
  paramCount() { return 2 * this.dModel; }
}

/**
 * Transformer Encoder Block
 * Self-attention + residual + layer norm + feedforward + residual + layer norm
 */
export class TransformerEncoderBlock {
  constructor(dModel, numHeads, dFF = null) {
    this.dModel = dModel;
    this.dFF = dFF || dModel * 4; // Standard transformer: 4x expansion
    this.outputSize = dModel;
    this.training = true;
    
    this.attention = new MultiHeadAttention(dModel, numHeads);
    this.norm1 = new LayerNorm(dModel);
    this.norm2 = new LayerNorm(dModel);
    
    // Feedforward: two Dense layers applied per-position
    this.ff1 = new Dense(dModel, this.dFF, 'relu');
    this.ff2 = new Dense(this.dFF, dModel, 'linear');
    
    // Cache
    this._attnInput = null;
    this._ffInput = null;
  }
  
  forward(input) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    
    // Self-attention + residual
    this._attnInput = input;
    const attnOut = this.attention.forward(input);
    const residual1 = addMatrices(input, attnOut);
    const normed1 = this.norm1.forward(residual1);
    
    // Feedforward per position + residual
    this._ffInput = normed1;
    const ffOut = new Matrix(batchSize, input.cols);
    
    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < seqLen; t++) {
        // Extract position vector: [1, dModel]
        const posVec = new Matrix(1, this.dModel);
        for (let d = 0; d < this.dModel; d++)
          posVec.set(0, d, normed1.get(b, t * this.dModel + d));
        
        // FF1 → ReLU → FF2
        const ff1Out = this.ff1.forward(posVec);
        const ff2Out = this.ff2.forward(ff1Out);
        
        for (let d = 0; d < this.dModel; d++)
          ffOut.set(b, t * this.dModel + d, ff2Out.get(0, d));
      }
    }
    
    const residual2 = addMatrices(normed1, ffOut);
    return this.norm2.forward(residual2);
  }
  
  backward(dOutput) {
    // Simplified backward through the block
    const dNorm2 = this.norm2.backward(dOutput);
    // Residual: gradient flows to both FF and skip connection
    const dFF = dNorm2; // simplified
    const dNorm1 = this.norm1.backward(dFF);
    const dAttn = this.attention.backward(dNorm1);
    return addMatrices(dAttn, dNorm1); // Residual gradient
  }
  
  update(learningRate) {
    this.attention.update(learningRate);
    this.norm1.update(learningRate);
    this.norm2.update(learningRate);
    this.ff1.update(learningRate, 0, 'sgd');
    this.ff2.update(learningRate, 0, 'sgd');
  }
  
  paramCount() {
    return this.attention.paramCount() + this.norm1.paramCount() + this.norm2.paramCount() +
           this.ff1.paramCount() + this.ff2.paramCount();
  }
}

function addMatrices(a, b) {
  const result = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++)
      result.set(i, j, a.get(i, j) + b.get(i, j));
  return result;
}
