// groupnorm.js — Group Normalization (Wu & He, 2018)
// Divides channels into groups and normalizes within each group.
// Unlike BatchNorm, GroupNorm doesn't depend on batch statistics.
// Unlike LayerNorm, GroupNorm normalizes within channel groups, not across all channels.
// 
// GroupNorm(1) = LayerNorm, GroupNorm(C) = InstanceNorm

import { Matrix } from './matrix.js';

export class GroupNorm {
  /**
   * @param {number} numGroups - Number of groups to divide channels into
   * @param {number} numChannels - Number of input channels (must be divisible by numGroups)
   * @param {number} eps - Small epsilon for numerical stability
   * @param {boolean} affine - Whether to learn scale (gamma) and shift (beta)
   */
  constructor(numGroups, numChannels, eps = 1e-5, affine = true) {
    if (numChannels % numGroups !== 0) {
      throw new Error(`numChannels (${numChannels}) must be divisible by numGroups (${numGroups})`);
    }
    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.channelsPerGroup = numChannels / numGroups;
    this.eps = eps;
    this.affine = affine;
    
    if (affine) {
      this.gamma = new Float64Array(numChannels).fill(1); // Scale
      this.beta = new Float64Array(numChannels).fill(0);  // Shift
      this.dGamma = new Float64Array(numChannels);
      this.dBeta = new Float64Array(numChannels);
    }
    
    // Saved for backward
    this._input = null;
    this._normalized = null;
    this._groupMeans = null;
    this._groupStds = null;
  }

  /**
   * Forward pass: normalize within each group.
   * @param {Matrix} x - Input (batchSize × numChannels)
   * @returns {Matrix} Normalized output
   */
  forward(x) {
    const B = x.rows;
    const C = x.cols;
    if (C !== this.numChannels) throw new Error(`Expected ${this.numChannels} channels, got ${C}`);
    
    this._input = x;
    const output = new Matrix(B, C);
    this._normalized = new Matrix(B, C);
    this._groupMeans = new Float64Array(B * this.numGroups);
    this._groupStds = new Float64Array(B * this.numGroups);
    
    for (let b = 0; b < B; b++) {
      for (let g = 0; g < this.numGroups; g++) {
        const start = g * this.channelsPerGroup;
        const end = start + this.channelsPerGroup;
        
        // Compute group mean
        let mean = 0;
        for (let c = start; c < end; c++) mean += x.get(b, c);
        mean /= this.channelsPerGroup;
        
        // Compute group variance
        let variance = 0;
        for (let c = start; c < end; c++) {
          const diff = x.get(b, c) - mean;
          variance += diff * diff;
        }
        variance /= this.channelsPerGroup;
        
        const std = Math.sqrt(variance + this.eps);
        this._groupMeans[b * this.numGroups + g] = mean;
        this._groupStds[b * this.numGroups + g] = std;
        
        // Normalize
        for (let c = start; c < end; c++) {
          const normalized = (x.get(b, c) - mean) / std;
          this._normalized.set(b, c, normalized);
          
          if (this.affine) {
            output.set(b, c, normalized * this.gamma[c] + this.beta[c]);
          } else {
            output.set(b, c, normalized);
          }
        }
      }
    }
    
    return output;
  }

  /**
   * Backward pass: compute gradients.
   * @param {Matrix} dOutput - Gradient of loss w.r.t. output
   * @returns {Matrix} Gradient w.r.t. input
   */
  backward(dOutput) {
    const B = dOutput.rows;
    const C = dOutput.cols;
    const dInput = new Matrix(B, C);
    
    // Gradient for gamma and beta
    if (this.affine) {
      this.dGamma.fill(0);
      this.dBeta.fill(0);
      for (let b = 0; b < B; b++) {
        for (let c = 0; c < C; c++) {
          this.dGamma[c] += dOutput.get(b, c) * this._normalized.get(b, c);
          this.dBeta[c] += dOutput.get(b, c);
        }
      }
    }
    
    // Gradient for input
    for (let b = 0; b < B; b++) {
      for (let g = 0; g < this.numGroups; g++) {
        const start = g * this.channelsPerGroup;
        const end = start + this.channelsPerGroup;
        const N = this.channelsPerGroup;
        const std = this._groupStds[b * this.numGroups + g];
        
        // Compute intermediate sums for the group
        let sumDy = 0, sumDyX = 0;
        for (let c = start; c < end; c++) {
          const dy = this.affine ? dOutput.get(b, c) * this.gamma[c] : dOutput.get(b, c);
          sumDy += dy;
          sumDyX += dy * this._normalized.get(b, c);
        }
        
        for (let c = start; c < end; c++) {
          const dy = this.affine ? dOutput.get(b, c) * this.gamma[c] : dOutput.get(b, c);
          const xHat = this._normalized.get(b, c);
          dInput.set(b, c, (dy - sumDy / N - xHat * sumDyX / N) / std);
        }
      }
    }
    
    return dInput;
  }

  update(lr) {
    if (!this.affine) return;
    for (let i = 0; i < this.numChannels; i++) {
      this.gamma[i] -= lr * this.dGamma[i];
      this.beta[i] -= lr * this.dBeta[i];
    }
  }

  paramCount() {
    return this.affine ? this.numChannels * 2 : 0;
  }
}
