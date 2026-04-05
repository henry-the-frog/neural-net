// embedding.js — Token Embedding layer

import { Matrix } from './matrix.js';

/**
 * Embedding layer — maps integer tokens to dense vectors
 * Input: [batch, seqLen] where values are token IDs (integers as floats)
 * Output: [batch, seqLen * embedDim]
 * 
 * This is the standard lookup table embedding used in NLP models.
 * Gradients update only the rows corresponding to tokens in the batch.
 */
export class Embedding {
  constructor(vocabSize, embedDim) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.outputSize = embedDim;
    this.training = true;
    
    // Initialize embedding matrix: [vocabSize, embedDim]
    // Normal initialization scaled by 1/sqrt(embedDim)
    const scale = 1 / Math.sqrt(embedDim);
    this.weights = Matrix.random(vocabSize, embedDim).mul(scale);
    
    // Cache
    this._tokenIds = null;
    this._batchSize = 0;
    this._seqLen = 0;
    
    // Gradients
    this.dWeights = null;
    this.dBiases = null;
  }
  
  forward(input) {
    // input: [batch, seqLen] — each value is a token ID
    this._batchSize = input.rows;
    this._seqLen = input.cols;
    this._tokenIds = [];
    
    const output = new Matrix(input.rows, input.cols * this.embedDim);
    
    for (let b = 0; b < input.rows; b++) {
      const batchTokens = [];
      for (let t = 0; t < input.cols; t++) {
        const tokenId = Math.round(input.get(b, t)); // Round to integer
        const clampedId = Math.max(0, Math.min(this.vocabSize - 1, tokenId));
        batchTokens.push(clampedId);
        
        // Lookup embedding vector
        for (let d = 0; d < this.embedDim; d++) {
          output.set(b, t * this.embedDim + d, this.weights.get(clampedId, d));
        }
      }
      this._tokenIds.push(batchTokens);
    }
    
    this.outputSize = this._seqLen * this.embedDim;
    return output;
  }
  
  backward(dOutput) {
    // Accumulate gradients for each token's embedding row
    this.dWeights = Matrix.zeros(this.vocabSize, this.embedDim);
    
    for (let b = 0; b < this._batchSize; b++) {
      for (let t = 0; t < this._seqLen; t++) {
        const tokenId = this._tokenIds[b][t];
        for (let d = 0; d < this.embedDim; d++) {
          const grad = dOutput.get(b, t * this.embedDim + d);
          this.dWeights.set(tokenId, d, this.dWeights.get(tokenId, d) + grad);
        }
      }
    }
    
    // Average over batch
    this.dWeights = this.dWeights.mul(1 / this._batchSize);
    this.dBiases = Matrix.zeros(1, 1); // Dummy for optimizer compatibility
    
    // No meaningful gradient for input (discrete tokens)
    return new Matrix(this._batchSize, this._seqLen);
  }
  
  update(learningRate) {
    this.weights = this.weights.sub(this.dWeights.mul(learningRate));
  }
  
  paramCount() {
    return this.vocabSize * this.embedDim;
  }
}
