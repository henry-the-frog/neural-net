// early-stopping.js — Early Stopping for training
export class EarlyStopping {
  constructor(options = {}) {
    // Support both positional (patience, minDelta) and object ({ patience, minDelta, mode })
    if (typeof options === 'number') {
      this.patience = options;
      this.minDelta = arguments[1] || 0;
      this.mode = 'min';
    } else {
      this.patience = options.patience || 5;
      this.minDelta = options.minDelta || 0;
      this.mode = options.mode || 'min';
    }
    this.bestScore = null;
    this.bestValue = null;
    this.bestEpoch = 0;
    this.counter = 0;
    this.shouldStop = false;
    this.stopped = false;
    this._epoch = 0;
  }

  step(score) {
    this._epoch++;
    const improved = this.mode === 'max'
      ? (this.bestScore === null || score > this.bestScore + this.minDelta)
      : (this.bestScore === null || score < this.bestScore - this.minDelta);
    
    if (improved) {
      this.bestScore = score;
      this.bestValue = score;
      this.bestEpoch = this._epoch;
      this.counter = 0;
    } else {
      this.counter++;
      if (this.counter >= this.patience) {
        this.shouldStop = true;
        this.stopped = true;
      }
    }
    return this.shouldStop;
  }

  summary() {
    return {
      stopped: this.stopped,
      bestValue: this.bestValue,
      bestEpoch: this.bestEpoch,
      totalEpochs: this._epoch,
      patience: this.patience,
    };
  }

  reset() {
    this.bestScore = null;
    this.bestValue = null;
    this.bestEpoch = 0;
    this.counter = 0;
    this.shouldStop = false;
    this.stopped = false;
    this._epoch = 0;
  }
}

/**
 * Train a model with early stopping based on validation loss.
 * @param {object} model - Network with train() and forward() methods
 * @param {Matrix} trainInputs - Training input data
 * @param {Matrix} trainTargets - Training target data
 * @param {Matrix} valInputs - Validation input data
 * @param {Matrix} valTargets - Validation target data
 * @param {object} options
 * @param {number} options.maxEpochs - Maximum epochs (default 100)
 * @param {number} options.lr - Learning rate (default 0.01)
 * @param {number} options.patience - Early stopping patience (default 10)
 * @param {number} options.minDelta - Minimum improvement (default 0)
 * @returns {{ totalEpochs: number, bestValue: number, history: number[] }}
 */
export function trainWithEarlyStopping(model, trainInputs, trainTargets, valInputs, valTargets, options = {}) {
  const {
    maxEpochs = 100,
    lr = 0.01,
    patience = 10,
    minDelta = 0,
  } = options;

  const stopper = new EarlyStopping({ patience, minDelta: minDelta || 1e-3 });
  const history = [];
  let totalEpochs = 0;

  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    // Train one epoch
    model.train({ inputs: trainInputs, targets: trainTargets }, {
      epochs: 1,
      learningRate: lr,
    });
    totalEpochs++;

    // Compute validation loss
    const valOutput = model.forward(valInputs);
    let valLoss = 0;
    for (let i = 0; i < valOutput.rows; i++) {
      for (let j = 0; j < valOutput.cols; j++) {
        const diff = valOutput.get(i, j) - valTargets.get(i, j);
        valLoss += diff * diff;
      }
    }
    valLoss /= (2 * valOutput.rows);
    history.push(valLoss);

    if (stopper.step(valLoss)) {
      break;
    }
  }

  return {
    totalEpochs,
    bestValue: stopper.bestScore,
    history,
  };
}
