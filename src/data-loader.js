// data-loader.js — Mini-batch Data Loader
// Shuffles, batches, and iterates over datasets.

export class DataLoader {
  constructor(data, batchSize = 32, shuffle = true) {
    this.data = data;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.indices = Array.from({ length: data.length }, (_, i) => i);
  }

  *[Symbol.iterator]() {
    if (this.shuffle) {
      for (let i = this.indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [this.indices[i], this.indices[j]] = [this.indices[j], this.indices[i]];
      }
    }

    for (let start = 0; start < this.indices.length; start += this.batchSize) {
      const batchIndices = this.indices.slice(start, start + this.batchSize);
      yield batchIndices.map(i => this.data[i]);
    }
  }

  get numBatches() {
    return Math.ceil(this.data.length / this.batchSize);
  }
}
