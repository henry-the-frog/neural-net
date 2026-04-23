// reward-model.js — Reward Model for RLHF (Ouyang et al., 2022)
// Maps (prompt, response) → scalar reward.
// Trained on human preference data using Bradley-Terry model.
//
// Architecture: Transformer encoder → mean pooling → linear → scalar

import { Matrix } from './matrix.js';

/**
 * Simple reward model: encodes text, pools, and produces a scalar reward.
 * For real RLHF, this would be a full transformer. Here we use a lightweight FFN.
 */
export class RewardModel {
  /**
   * @param {number} inputDim - Dimension of text encoding
   * @param {number} hiddenDim - FFN hidden dimension
   */
  constructor(inputDim, hiddenDim = 64) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    
    // FFN: input → hidden → 1
    this.W1 = Matrix.random(inputDim, hiddenDim).map(v => v * Math.sqrt(2.0 / inputDim));
    this.b1 = new Float64Array(hiddenDim);
    this.W2 = Matrix.random(hiddenDim, 1).map(v => v * Math.sqrt(2.0 / hiddenDim));
    this.b2 = new Float64Array(1);
    
    // Gradients
    this.dW1 = null;
    this.db1 = null;
    this.dW2 = null;
    this.db2 = null;
    
    this._input = null;
    this._hidden = null;
  }

  /**
   * Forward: encoding → reward scalar.
   * @param {Float64Array} encoding - Mean-pooled text encoding
   * @returns {number} Reward scalar
   */
  forward(encoding) {
    this._input = encoding;
    
    // Hidden = ReLU(encoding @ W1 + b1)
    this._hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputDim; i++) {
        sum += encoding[i] * this.W1.get(i, j);
      }
      this._hidden[j] = Math.max(0, sum); // ReLU
    }
    
    // Reward = hidden @ W2 + b2
    let reward = this.b2[0];
    for (let j = 0; j < this.hiddenDim; j++) {
      reward += this._hidden[j] * this.W2.get(j, 0);
    }
    
    return reward;
  }

  /**
   * Compute Bradley-Terry preference loss.
   * Given (chosen, rejected) encoding pairs:
   * L = -log(σ(r_chosen - r_rejected))
   * 
   * @param {Array<{chosen: Float64Array, rejected: Float64Array}>} pairs
   * @returns {{ loss: number, accuracy: number }}
   */
  preferenceLoss(pairs) {
    let totalLoss = 0;
    let correct = 0;
    
    for (const { chosen, rejected } of pairs) {
      const rChosen = this.forward(chosen);
      const rRejected = this.forward(rejected);
      
      const diff = rChosen - rRejected;
      // -log(σ(diff)) = log(1 + exp(-diff))
      const loss = diff >= 0 ? Math.log(1 + Math.exp(-diff)) : -diff + Math.log(1 + Math.exp(diff));
      totalLoss += loss;
      
      if (rChosen > rRejected) correct++;
    }
    
    return {
      loss: totalLoss / pairs.length,
      accuracy: correct / pairs.length,
    };
  }

  /**
   * Train on preference pairs for one epoch.
   */
  trainStep(pairs, lr = 0.001) {
    const { loss, accuracy } = this.preferenceLoss(pairs);
    
    // Simple gradient descent on preference loss
    // For each pair, compute gradient of -log σ(r_w - r_l)
    this.dW1 = new Matrix(this.inputDim, this.hiddenDim);
    this.db1 = new Float64Array(this.hiddenDim);
    this.dW2 = new Matrix(this.hiddenDim, 1);
    this.db2 = new Float64Array(1);
    
    for (const { chosen, rejected } of pairs) {
      const rChosen = this.forward(chosen);
      const chosenHidden = new Float64Array(this._hidden);
      const chosenInput = new Float64Array(this._input);
      
      const rRejected = this.forward(rejected);
      const rejectedHidden = new Float64Array(this._hidden);
      const rejectedInput = new Float64Array(this._input);
      
      const diff = rChosen - rRejected;
      const grad = -1 / (1 + Math.exp(diff)); // gradient of -log σ(diff) w.r.t. diff
      
      // Gradient flows through both chosen (+) and rejected (-)
      for (let j = 0; j < this.hiddenDim; j++) {
        const dReward_w = chosenHidden[j] > 0 ? 1 : 0;
        const dReward_l = rejectedHidden[j] > 0 ? 1 : 0;
        
        this.dW2.set(j, 0, this.dW2.get(j, 0) + grad * (chosenHidden[j] - rejectedHidden[j]));
        
        for (let i = 0; i < this.inputDim; i++) {
          const dChosen = grad * this.W2.get(j, 0) * dReward_w * chosenInput[i];
          const dRejected = -grad * this.W2.get(j, 0) * dReward_l * rejectedInput[i];
          this.dW1.set(i, j, this.dW1.get(i, j) + dChosen + dRejected);
        }
      }
    }
    
    // Scale gradients
    const scale = 1 / pairs.length;
    for (let i = 0; i < this.dW1.data.length; i++) this.dW1.data[i] *= scale;
    for (let i = 0; i < this.dW2.data.length; i++) this.dW2.data[i] *= scale;
    
    // Update
    for (let i = 0; i < this.W1.data.length; i++) this.W1.data[i] -= lr * this.dW1.data[i];
    for (let i = 0; i < this.W2.data.length; i++) this.W2.data[i] -= lr * this.dW2.data[i];
    
    return { loss, accuracy };
  }

  paramCount() {
    return this.inputDim * this.hiddenDim + this.hiddenDim + this.hiddenDim * 1 + 1;
  }
}
