// history.js — Training history tracking and visualization
// Records metrics per epoch and provides ASCII loss plots

export class TrainingHistory {
  constructor() {
    this.epochs = [];
    this.startTime = Date.now();
  }

  record(epoch, metrics = {}) {
    this.epochs.push({
      epoch,
      time: Date.now() - this.startTime,
      ...metrics
    });
  }

  get length() { return this.epochs.length; }
  get losses() { return this.epochs.map(e => e.loss); }
  get lrs() { return this.epochs.map(e => e.lr); }

  last() { return this.epochs[this.epochs.length - 1]; }

  // Best epoch by lowest loss
  best() {
    let best = this.epochs[0];
    for (const e of this.epochs) {
      if (e.loss < best.loss) best = e;
    }
    return best;
  }

  // Summary stats
  summary() {
    const losses = this.losses;
    return {
      epochs: losses.length,
      initialLoss: losses[0],
      finalLoss: losses[losses.length - 1],
      bestLoss: Math.min(...losses),
      bestEpoch: losses.indexOf(Math.min(...losses)),
      improvement: ((losses[0] - losses[losses.length - 1]) / losses[0] * 100).toFixed(1) + '%',
      totalTimeMs: this.epochs[this.epochs.length - 1]?.time || 0,
    };
  }

  /**
   * ASCII loss plot
   * @param {number} width - Chart width in chars
   * @param {number} height - Chart height in lines
   * @returns {string}
   */
  plotLoss(width = 60, height = 15) {
    const losses = this.losses;
    if (losses.length === 0) return '(no data)';

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    // Downsample if needed
    const step = Math.max(1, Math.floor(losses.length / width));
    const points = [];
    for (let i = 0; i < losses.length; i += step) {
      points.push(losses[i]);
    }

    const lines = [];
    lines.push(`Loss: ${maxLoss.toFixed(4)} ┐`);

    for (let row = height - 1; row >= 0; row--) {
      const threshold = minLoss + (range * (row + 0.5)) / height;
      let line = '│';
      for (let col = 0; col < points.length && col < width; col++) {
        if (Math.abs(points[col] - threshold) < range / height) {
          line += '●';
        } else if (points[col] > threshold) {
          line += ' ';
        } else {
          line += ' ';
        }
      }
      lines.push(line);
    }

    lines.push(`       ${minLoss.toFixed(4)} ┘${'─'.repeat(Math.min(points.length, width))}`);
    lines.push(`       Epoch 0${''.padEnd(Math.min(points.length, width) - 10)}${losses.length - 1}`);

    return lines.join('\n');
  }

  /**
   * Simple sparkline of loss values
   * @returns {string}
   */
  sparkline() {
    const chars = '▁▂▃▄▅▆▇█';
    const losses = this.losses;
    if (losses.length === 0) return '';

    const min = Math.min(...losses);
    const max = Math.max(...losses);
    const range = max - min || 1;

    // Downsample to ~40 chars
    const step = Math.max(1, Math.floor(losses.length / 40));
    let result = '';
    for (let i = 0; i < losses.length; i += step) {
      const idx = Math.min(chars.length - 1, Math.floor((losses[i] - min) / range * (chars.length - 1)));
      result += chars[idx];
    }
    return result;
  }
}
