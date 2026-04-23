// early-stopping.js — Early Stopping for training
export class EarlyStopping {
  constructor(patience = 5, minDelta = 0) {
    this.patience = patience;
    this.minDelta = minDelta;
    this.bestScore = null;
    this.counter = 0;
    this.shouldStop = false;
  }

  step(score) {
    if (this.bestScore === null || score < this.bestScore - this.minDelta) {
      this.bestScore = score;
      this.counter = 0;
    } else {
      this.counter++;
      if (this.counter >= this.patience) this.shouldStop = true;
    }
    return this.shouldStop;
  }

  reset() {
    this.bestScore = null;
    this.counter = 0;
    this.shouldStop = false;
  }
}
