// data-loader.js — Mini-batch Data Loader
// Shuffles, batches, and iterates over datasets.
// Supports both raw arrays and Matrix-based datasets.

import { Matrix } from './matrix.js';

export class DataLoader {
  /**
   * @param {Array|{inputs: Matrix, targets: Matrix}} data - Array of items or {inputs, targets}
   * @param {number} batchSize
   * @param {boolean} shuffle
   */
  constructor(data, batchSize = 32, shuffle = true) {
    if (data && data.inputs instanceof Matrix) {
      this._mode = 'matrix';
      this._inputs = data.inputs;
      this._targets = data.targets;
      this._length = data.inputs.rows;
    } else {
      this._mode = 'array';
      this.data = data;
      this._length = data.length;
    }
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.indices = Array.from({ length: this._length }, (_, i) => i);
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

      if (this._mode === 'matrix') {
        const batchInputs = new Matrix(batchIndices.length, this._inputs.cols);
        const batchTargets = new Matrix(batchIndices.length, this._targets.cols);
        for (let i = 0; i < batchIndices.length; i++) {
          const idx = batchIndices[i];
          for (let j = 0; j < this._inputs.cols; j++) batchInputs.set(i, j, this._inputs.get(idx, j));
          for (let j = 0; j < this._targets.cols; j++) batchTargets.set(i, j, this._targets.get(idx, j));
        }
        yield { inputs: batchInputs, targets: batchTargets };
      } else {
        yield batchIndices.map(i => this.data[i]);
      }
    }
  }

  get numBatches() {
    return Math.ceil(this._length / this.batchSize);
  }

  get length() {
    return this._length;
  }
}

/**
 * Split dataset into train/validation/test sets.
 * @param {Matrix} inputs
 * @param {Matrix} targets
 * @param {Object} options
 * @param {number} options.valRatio - Validation set ratio (0-1), default 0.1
 * @param {number} options.testRatio - Test set ratio (0-1), default 0.1
 * @param {boolean} options.shuffle - Shuffle before splitting, default true
 * @returns {{train: {inputs, targets}, val: {inputs, targets}, test: {inputs, targets}}}
 */
export function trainValTestSplit(inputs, targets, { valRatio = 0.1, testRatio = 0.1, shuffle: doShuffle = true } = {}) {
  const n = inputs.rows;
  const indices = Array.from({ length: n }, (_, i) => i);

  if (doShuffle) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }

  const testSize = Math.floor(n * testRatio);
  const valSize = Math.floor(n * valRatio);
  const trainSize = n - testSize - valSize;

  const sliceMatrix = (mat, idxs) => {
    const result = new Matrix(idxs.length, mat.cols);
    for (let i = 0; i < idxs.length; i++) {
      for (let j = 0; j < mat.cols; j++) result.set(i, j, mat.get(idxs[i], j));
    }
    return result;
  };

  const trainIdx = indices.slice(0, trainSize);
  const valIdx = indices.slice(trainSize, trainSize + valSize);
  const testIdx = indices.slice(trainSize + valSize);

  return {
    train: { inputs: sliceMatrix(inputs, trainIdx), targets: sliceMatrix(targets, trainIdx) },
    val: { inputs: sliceMatrix(inputs, valIdx), targets: sliceMatrix(targets, valIdx) },
    test: { inputs: sliceMatrix(inputs, testIdx), targets: sliceMatrix(targets, testIdx) },
  };
}

/**
 * Stratified split — preserves class distribution in each split.
 * Targets must be single-column (class labels).
 * @param {Matrix} inputs
 * @param {Matrix} targets - Nx1 matrix of integer class labels
 * @param {Object} options
 * @param {number} options.valRatio
 * @param {number} options.testRatio
 * @returns {{train: {inputs, targets}, val: {inputs, targets}, test: {inputs, targets}}}
 */
export function stratifiedSplit(inputs, targets, { valRatio = 0.1, testRatio = 0.1 } = {}) {
  // Group indices by class
  const classIndices = {};
  for (let i = 0; i < targets.rows; i++) {
    const cls = targets.get(i, 0);
    if (!classIndices[cls]) classIndices[cls] = [];
    classIndices[cls].push(i);
  }

  const trainIdx = [], valIdx = [], testIdx = [];

  for (const cls of Object.keys(classIndices)) {
    const idxs = classIndices[cls];
    // Shuffle within class
    for (let i = idxs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idxs[i], idxs[j]] = [idxs[j], idxs[i]];
    }

    const testN = Math.max(1, Math.round(idxs.length * testRatio));
    const valN = Math.max(1, Math.round(idxs.length * valRatio));
    const trainN = idxs.length - testN - valN;

    trainIdx.push(...idxs.slice(0, trainN));
    valIdx.push(...idxs.slice(trainN, trainN + valN));
    testIdx.push(...idxs.slice(trainN + valN));
  }

  const sliceMatrix = (mat, idxs) => {
    const result = new Matrix(idxs.length, mat.cols);
    for (let i = 0; i < idxs.length; i++) {
      for (let j = 0; j < mat.cols; j++) result.set(i, j, mat.get(idxs[i], j));
    }
    return result;
  };

  return {
    train: { inputs: sliceMatrix(inputs, trainIdx), targets: sliceMatrix(targets, trainIdx) },
    val: { inputs: sliceMatrix(inputs, valIdx), targets: sliceMatrix(targets, valIdx) },
    test: { inputs: sliceMatrix(inputs, testIdx), targets: sliceMatrix(targets, testIdx) },
  };
}

/**
 * K-fold cross-validation generator.
 * @param {Matrix} inputs
 * @param {Matrix} targets
 * @param {number} k - Number of folds
 * @yields {{train: {inputs, targets}, val: {inputs, targets}, fold: number}}
 */
export function* kFoldSplit(inputs, targets, k = 5) {
  const n = inputs.rows;
  const indices = Array.from({ length: n }, (_, i) => i);
  // Shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const foldSize = Math.floor(n / k);
  const sliceMatrix = (mat, idxs) => {
    const result = new Matrix(idxs.length, mat.cols);
    for (let i = 0; i < idxs.length; i++) {
      for (let j = 0; j < mat.cols; j++) result.set(i, j, mat.get(idxs[i], j));
    }
    return result;
  };

  for (let fold = 0; fold < k; fold++) {
    const valStart = fold * foldSize;
    const valEnd = fold === k - 1 ? n : valStart + foldSize;
    const valIdx = indices.slice(valStart, valEnd);
    const trainIdx = [...indices.slice(0, valStart), ...indices.slice(valEnd)];

    yield {
      fold,
      train: { inputs: sliceMatrix(inputs, trainIdx), targets: sliceMatrix(targets, trainIdx) },
      val: { inputs: sliceMatrix(inputs, valIdx), targets: sliceMatrix(targets, valIdx) },
    };
  }
}
