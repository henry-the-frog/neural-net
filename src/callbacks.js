// callbacks.js — Training callbacks (EarlyStopping, etc.)

/**
 * Early stopping — monitors a metric and stops training when it stops improving.
 * 
 * Usage:
 *   const es = new EarlyStopping({ patience: 10, minDelta: 1e-4 });
 *   net.train(data, { callbacks: [es] });
 */
export class EarlyStopping {
  constructor({
    patience = 10,
    minDelta = 0,
    mode = 'min',       // 'min' for loss, 'max' for accuracy
    restore = true,     // restore best weights on stop
    verbose = false
  } = {}) {
    this.patience = patience;
    this.minDelta = minDelta;
    this.mode = mode;
    this.restore = restore;
    this.verbose = verbose;
    this.bestValue = mode === 'min' ? Infinity : -Infinity;
    this.bestEpoch = 0;
    this.wait = 0;
    this.stopped = false;
    this.stoppedEpoch = 0;
    this._bestWeights = null;
  }

  _isBetter(current) {
    if (this.mode === 'min') {
      return current < this.bestValue - this.minDelta;
    }
    return current > this.bestValue + this.minDelta;
  }

  onEpochEnd(epoch, loss, network) {
    if (this._isBetter(loss)) {
      this.bestValue = loss;
      this.bestEpoch = epoch;
      this.wait = 0;
      // Save best weights
      if (this.restore && network) {
        this._bestWeights = network.toJSON();
      }
    } else {
      this.wait++;
      if (this.wait >= this.patience) {
        this.stopped = true;
        this.stoppedEpoch = epoch;
        if (this.verbose) {
          console.log(`Early stopping at epoch ${epoch + 1}. Best: ${this.bestValue.toFixed(6)} at epoch ${this.bestEpoch + 1}`);
        }
        // Restore best weights
        if (this.restore && this._bestWeights && network) {
          const restored = network.constructor.fromJSON(this._bestWeights);
          for (let i = 0; i < network.layers.length; i++) {
            if (restored.layers[i] && restored.layers[i].weights) {
              network.layers[i].weights = restored.layers[i].weights;
              network.layers[i].biases = restored.layers[i].biases;
            }
          }
        }
        return true; // signal to stop
      }
    }
    return false;
  }

  reset() {
    this.bestValue = this.mode === 'min' ? Infinity : -Infinity;
    this.bestEpoch = 0;
    this.wait = 0;
    this.stopped = false;
    this.stoppedEpoch = 0;
    this._bestWeights = null;
  }
}

/**
 * LossHistory callback — records per-epoch loss values.
 */
export class LossHistory {
  constructor() {
    this.losses = [];
  }

  onEpochEnd(epoch, loss) {
    this.losses.push(loss);
    return false;
  }

  reset() {
    this.losses = [];
  }
}
