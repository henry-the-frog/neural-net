// training-logger.js — Training metrics logger for neural networks
//
// Usage:
//   const logger = new TrainingLogger();
//   for (let epoch = 0; epoch < 100; epoch++) {
//     const loss = net.trainBatch(x, y, lr);
//     logger.log({ epoch, loss });
//   }
//   logger.summary();     // Print training summary
//   logger.toJSON();      // Export as JSON
//   logger.toCSV();       // Export as CSV

/**
 * Training metric logger with statistics and export.
 */
export class TrainingLogger {
  constructor(name = 'training') {
    this.name = name;
    this.entries = [];
    this.startTime = Date.now();
    this.checkpoints = [];
  }

  /**
   * Log metrics for an epoch/step.
   * @param {Object} metrics - { epoch, loss, accuracy, lr, ... }
   */
  log(metrics) {
    this.entries.push({
      ...metrics,
      timestamp: Date.now() - this.startTime,
    });
    return this;
  }

  /**
   * Save a checkpoint (named snapshot of current state).
   */
  checkpoint(name) {
    this.checkpoints.push({
      name,
      entry: this.entries.length - 1,
      timestamp: Date.now() - this.startTime,
    });
    return this;
  }

  /**
   * Get the best entry by a metric (default: lowest loss).
   */
  best(metric = 'loss', mode = 'min') {
    if (this.entries.length === 0) return null;
    let bestIdx = 0;
    for (let i = 1; i < this.entries.length; i++) {
      const better = mode === 'min'
        ? this.entries[i][metric] < this.entries[bestIdx][metric]
        : this.entries[i][metric] > this.entries[bestIdx][metric];
      if (better) bestIdx = i;
    }
    return this.entries[bestIdx];
  }

  /**
   * Get the last N entries.
   */
  tail(n = 5) {
    return this.entries.slice(-n);
  }

  /**
   * Calculate statistics for a metric.
   */
  stats(metric = 'loss') {
    const values = this.entries.map(e => e[metric]).filter(v => v !== undefined);
    if (values.length === 0) return null;
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const last = values[values.length - 1];
    const first = values[0];
    
    return { min, max, mean, first, last, count: values.length };
  }

  /**
   * Check if training is improving (loss decreasing over last N entries).
   */
  isImproving(metric = 'loss', window = 10, mode = 'min') {
    if (this.entries.length < window * 2) return true; // Not enough data
    
    const recent = this.entries.slice(-window);
    const older = this.entries.slice(-window * 2, -window);
    
    const recentAvg = recent.reduce((s, e) => s + (e[metric] || 0), 0) / window;
    const olderAvg = older.reduce((s, e) => s + (e[metric] || 0), 0) / window;
    
    return mode === 'min' ? recentAvg < olderAvg : recentAvg > olderAvg;
  }

  /**
   * Get training summary.
   */
  summary() {
    const lossStats = this.stats('loss');
    const elapsed = Date.now() - this.startTime;
    
    return {
      name: this.name,
      epochs: this.entries.length,
      elapsed_ms: elapsed,
      loss: lossStats,
      best: this.best(),
      improving: this.isImproving(),
      checkpoints: this.checkpoints.map(c => c.name),
    };
  }

  /**
   * Export to JSON string.
   */
  toJSON() {
    return JSON.stringify({
      name: this.name,
      entries: this.entries,
      checkpoints: this.checkpoints,
      summary: this.summary(),
    }, null, 2);
  }

  /**
   * Export to CSV string.
   */
  toCSV() {
    if (this.entries.length === 0) return '';
    const keys = Object.keys(this.entries[0]);
    const header = keys.join(',');
    const rows = this.entries.map(e => keys.map(k => e[k] ?? '').join(','));
    return [header, ...rows].join('\n');
  }

  /**
   * Create an ASCII loss chart.
   */
  chart(metric = 'loss', width = 60, height = 15) {
    const values = this.entries.map(e => e[metric]).filter(v => v !== undefined);
    if (values.length === 0) return 'No data';
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    
    // Downsample if needed
    const step = Math.max(1, Math.floor(values.length / width));
    const sampled = [];
    for (let i = 0; i < values.length; i += step) {
      sampled.push(values[i]);
    }
    
    const lines = [];
    for (let row = 0; row < height; row++) {
      const threshold = max - (row / (height - 1)) * range;
      let line = '';
      for (const v of sampled) {
        line += v >= threshold ? '█' : ' ';
      }
      const label = threshold.toFixed(4).padStart(10);
      lines.push(`${label} │${line}`);
    }
    lines.push(`${''.padStart(10)} └${'─'.repeat(sampled.length)}`);
    return lines.join('\n');
  }
}

/**
 * Convenience: create a logger and wrap a training loop.
 */
export function trainWithLogging(net, inputs, targets, {
  epochs = 100,
  lr = 0.01,
  logEvery = 1,
  name = 'training',
} = {}) {
  const logger = new TrainingLogger(name);
  
  for (let epoch = 0; epoch < epochs; epoch++) {
    const loss = net.trainBatch(inputs, targets, lr);
    if (epoch % logEvery === 0) {
      logger.log({ epoch, loss });
    }
  }
  
  return logger;
}
