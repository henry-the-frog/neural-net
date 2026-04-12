// early-stopping.js — Early stopping callback for training
//
// Usage:
//   const stopper = new EarlyStopping({ patience: 10, minDelta: 0.001 });
//   for (let epoch = 0; epoch < maxEpochs; epoch++) {
//     const valLoss = evaluate(model, valData);
//     if (stopper.step(valLoss)) break; // Stop if no improvement
//   }
//   stopper.restore(model); // Restore best weights

import { snapshotWeights, restoreWeights } from './lottery-ticket.js';

/**
 * Early stopping monitor.
 * Stops training when validation loss stops improving.
 */
export class EarlyStopping {
  constructor({
    patience = 10,      // Number of epochs to wait for improvement
    minDelta = 0,       // Minimum improvement to count as progress
    mode = 'min',       // 'min' (lower is better) or 'max' (higher is better)
    saveBest = true,    // Save best model weights
  } = {}) {
    this.patience = patience;
    this.minDelta = minDelta;
    this.mode = mode;
    this.saveBest = saveBest;
    
    this.bestValue = mode === 'min' ? Infinity : -Infinity;
    this.bestEpoch = 0;
    this.bestWeights = null;
    this.waitCount = 0;
    this.epoch = 0;
    this.stopped = false;
    this.history = [];
  }

  /**
   * Record a metric value and check if training should stop.
   * @param {number} value - Current metric value (e.g., validation loss)
   * @param {Network} [model] - Network to save (if saveBest=true)
   * @returns {boolean} true if training should stop
   */
  step(value, model = null) {
    this.epoch++;
    this.history.push(value);
    
    const improved = this.mode === 'min'
      ? value < this.bestValue - this.minDelta
      : value > this.bestValue + this.minDelta;
    
    if (improved) {
      this.bestValue = value;
      this.bestEpoch = this.epoch;
      this.waitCount = 0;
      if (this.saveBest && model) {
        this.bestWeights = snapshotWeights(model);
      }
    } else {
      this.waitCount++;
    }
    
    if (this.waitCount >= this.patience) {
      this.stopped = true;
      return true;
    }
    
    return false;
  }

  /**
   * Restore best weights to the model.
   */
  restore(model) {
    if (this.bestWeights) {
      restoreWeights(model, this.bestWeights);
    }
  }

  /**
   * Get summary of early stopping state.
   */
  summary() {
    return {
      stopped: this.stopped,
      bestValue: this.bestValue,
      bestEpoch: this.bestEpoch,
      totalEpochs: this.epoch,
      patienceUsed: this.waitCount,
    };
  }
}

/**
 * Convenience: train with early stopping.
 */
export function trainWithEarlyStopping(model, trainInputs, trainTargets, valInputs, valTargets, {
  maxEpochs = 1000,
  lr = 0.01,
  patience = 20,
  minDelta = 0.001,
} = {}) {
  const stopper = new EarlyStopping({ patience, minDelta, saveBest: true });
  
  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    model.trainBatch(trainInputs, trainTargets, lr);
    
    // Compute validation loss
    const pred = model.predict(valInputs);
    let valLoss = 0;
    for (let i = 0; i < valInputs.rows; i++) {
      for (let j = 0; j < valTargets.cols; j++) {
        const d = pred.get(i, j) - valTargets.get(i, j);
        valLoss += d * d;
      }
    }
    valLoss /= valInputs.rows;
    
    if (stopper.step(valLoss, model)) break;
  }
  
  stopper.restore(model);
  return stopper.summary();
}
