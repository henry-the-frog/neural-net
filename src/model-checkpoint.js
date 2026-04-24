// model-checkpoint.js — Model checkpointing and training resumption
// Save best model during training, resume from saved state.

/**
 * ModelCheckpoint callback — saves model state when metric improves.
 * Unlike EarlyStopping, this doesn't stop training — it just records
 * checkpoints that can be retrieved later.
 * 
 * Usage:
 *   const ckpt = new ModelCheckpoint({ metric: 'loss', mode: 'min' });
 *   net.train(data, { callbacks: [ckpt] });
 *   const bestModel = ckpt.getBestModel();
 *   const history = ckpt.getHistory();
 */
export class ModelCheckpoint {
  constructor({
    mode = 'min',       // 'min' for loss, 'max' for accuracy
    maxCheckpoints = 3, // Keep N best checkpoints
    verbose = false,
  } = {}) {
    this.mode = mode;
    this.maxCheckpoints = maxCheckpoints;
    this.verbose = verbose;
    this.checkpoints = [];  // sorted best→worst
    this._history = [];     // all epoch metrics
  }

  _isBetter(a, b) {
    return this.mode === 'min' ? a < b : a > b;
  }

  onEpochEnd(epoch, metric, network) {
    this._history.push({ epoch, metric });
    
    // Check if this is among the best
    const checkpoint = {
      epoch,
      metric,
      state: network.toJSON(),
      timestamp: Date.now(),
    };

    if (this.checkpoints.length < this.maxCheckpoints) {
      this.checkpoints.push(checkpoint);
      this.checkpoints.sort((a, b) => 
        this.mode === 'min' ? a.metric - b.metric : b.metric - a.metric
      );
      if (this.verbose) {
        console.log(`Checkpoint saved: epoch ${epoch + 1}, metric: ${metric.toFixed(6)}`);
      }
    } else if (this._isBetter(metric, this.checkpoints[this.checkpoints.length - 1].metric)) {
      this.checkpoints[this.checkpoints.length - 1] = checkpoint;
      this.checkpoints.sort((a, b) =>
        this.mode === 'min' ? a.metric - b.metric : b.metric - a.metric
      );
      if (this.verbose) {
        console.log(`Checkpoint updated: epoch ${epoch + 1}, metric: ${metric.toFixed(6)}`);
      }
    }

    return false; // never stop training
  }

  /** Get the best checkpoint's model state (JSON) */
  getBestModel() {
    return this.checkpoints.length > 0 ? this.checkpoints[0].state : null;
  }

  /** Get the best metric value */
  getBestMetric() {
    return this.checkpoints.length > 0 ? this.checkpoints[0].metric : null;
  }

  /** Get the best epoch number */
  getBestEpoch() {
    return this.checkpoints.length > 0 ? this.checkpoints[0].epoch : null;
  }

  /** Get all checkpoints (best first) */
  getCheckpoints() {
    return this.checkpoints.map(c => ({ epoch: c.epoch, metric: c.metric, timestamp: c.timestamp }));
  }

  /** Get full training history */
  getHistory() {
    return this._history;
  }

  reset() {
    this.checkpoints = [];
    this._history = [];
  }
}

/**
 * TrainingState — captures everything needed to resume training.
 * Includes model weights, optimizer state, epoch, and training config.
 */
export class TrainingState {
  /**
   * Capture the current state of training.
   * @param {Network} network
   * @param {Object} config — training config (epochs, lr, etc.)
   * @param {number} epoch — current epoch
   * @param {number[]} history — loss history
   * @param {Object} [optimizerState] — optimizer internal state
   */
  static capture(network, { epoch, history, config, optimizerState = null }) {
    return {
      version: 1,
      model: network.toJSON(),
      epoch,
      history,
      config,
      optimizerState,
      timestamp: Date.now(),
    };
  }

  /**
   * Resume training from a saved state.
   * @param {Function} NetworkClass — Network constructor (for fromJSON)
   * @param {Object} state — previously captured state
   * @param {Object} data — training data { inputs, targets }
   * @param {Object} [overrides] — override any training config
   * @returns {{ network, history }} — resumed network + full history
   */
  static resume(NetworkClass, state, data, overrides = {}) {
    const network = NetworkClass.fromJSON(state.model);
    const remainingEpochs = (overrides.epochs || state.config.epochs) - state.epoch;
    
    if (remainingEpochs <= 0) {
      return { network, history: state.history };
    }

    const config = {
      ...state.config,
      ...overrides,
      epochs: remainingEpochs,
    };

    const newHistory = network.train(data, config);
    return {
      network,
      history: [...state.history, ...newHistory],
      totalEpochs: state.epoch + newHistory.length,
    };
  }
}

/**
 * ReduceLROnPlateau callback — reduces learning rate when metric plateaus.
 * Requires the training loop to use the callback's lr property.
 * 
 * Usage:
 *   const scheduler = new ReduceLROnPlateau({ factor: 0.5, patience: 5 });
 *   net.train(data, { callbacks: [scheduler] });
 */
export class ReduceLROnPlateau {
  constructor({
    mode = 'min',
    factor = 0.5,
    patience = 5,
    minLR = 1e-6,
    verbose = false,
  } = {}) {
    this.mode = mode;
    this.factor = factor;
    this.patience = patience;
    this.minLR = minLR;
    this.verbose = verbose;
    this.bestValue = mode === 'min' ? Infinity : -Infinity;
    this.wait = 0;
    this.reductions = 0;
    this.currentLR = null;
  }

  _isBetter(current) {
    return this.mode === 'min' ? current < this.bestValue : current > this.bestValue;
  }

  onEpochEnd(epoch, metric, network) {
    if (this._isBetter(metric)) {
      this.bestValue = metric;
      this.wait = 0;
    } else {
      this.wait++;
      if (this.wait >= this.patience) {
        this.reductions++;
        this.wait = 0;
        if (this.verbose) {
          console.log(`ReduceLROnPlateau: reducing lr by factor ${this.factor} at epoch ${epoch + 1}`);
        }
      }
    }
    return false;
  }

  /** Get the effective learning rate multiplier */
  getLRMultiplier() {
    return Math.max(this.factor ** this.reductions, this.minLR);
  }

  reset() {
    this.bestValue = this.mode === 'min' ? Infinity : -Infinity;
    this.wait = 0;
    this.reductions = 0;
  }
}
