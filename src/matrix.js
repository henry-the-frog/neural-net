// matrix.js — Matrix operations for neural networks
// Flat Float64Array storage for performance

export class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    this.data = data || new Float64Array(rows * cols);
  }

  // Element access
  get(r, c) { return this.data[r * this.cols + c]; }
  set(r, c, v) { this.data[r * this.cols + c] = v; }

  // Fill with value
  fill(v) { this.data.fill(v); return this; }

  // Fill with random values (Xavier/Glorot initialization)
  randomize(scale = null) {
    const s = scale || Math.sqrt(2.0 / (this.rows + this.cols));
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = (Math.random() * 2 - 1) * s;
    }
    return this;
  }

  // Matrix multiplication: this × other
  dot(other) {
    if (this.cols !== other.rows) throw new Error(`Shape mismatch: ${this.rows}×${this.cols} · ${other.rows}×${other.cols}`);
    const result = new Matrix(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.data[i * this.cols + k] * other.data[k * other.cols + j];
        }
        result.data[i * other.cols + j] = sum;
      }
    }
    return result;
  }

  // Transpose
  transpose() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j * this.rows + i] = this.data[i * this.cols + j];
      }
    }
    return result;
  }

  // Element-wise addition
  add(other) {
    if (other instanceof Matrix) {
      if (this.rows !== other.rows || this.cols !== other.cols) {
        // Broadcasting: if other is a column vector (n×1), broadcast across columns
        if (other.cols === 1 && this.rows === other.rows) {
          const result = new Matrix(this.rows, this.cols);
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              result.data[i * this.cols + j] = this.data[i * this.cols + j] + other.data[i];
            }
          }
          return result;
        }
        // Broadcasting: if other is a row vector (1×n), broadcast across rows
        if (other.rows === 1 && this.cols === other.cols) {
          const result = new Matrix(this.rows, this.cols);
          for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              result.data[i * this.cols + j] = this.data[i * this.cols + j] + other.data[j];
            }
          }
          return result;
        }
        throw new Error(`Shape mismatch: ${this.rows}×${this.cols} + ${other.rows}×${other.cols}`);
      }
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] + other.data[i];
      return result;
    }
    // Scalar
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] + other;
    return result;
  }

  // Element-wise subtraction
  sub(other) {
    const result = new Matrix(this.rows, this.cols);
    if (other instanceof Matrix) {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] - other.data[i];
    } else {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] - other;
    }
    return result;
  }

  // Element-wise multiplication (Hadamard product)
  mul(other) {
    const result = new Matrix(this.rows, this.cols);
    if (other instanceof Matrix) {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] * other.data[i];
    } else {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] * other;
    }
    return result;
  }

  // Transpose
  T() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j * this.rows + i] = this.data[i * this.cols + j];
      }
    }
    return result;
  }

  // Apply function element-wise
  map(fn) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) result.data[i] = fn(this.data[i], i);
    return result;
  }

  // Sum all elements
  sum() {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s;
  }

  // Sum along axis (0 = sum rows → 1×cols, 1 = sum cols → rows×1)
  sumAxis(axis) {
    if (axis === 0) {
      const result = new Matrix(1, this.cols);
      for (let j = 0; j < this.cols; j++) {
        let s = 0;
        for (let i = 0; i < this.rows; i++) s += this.data[i * this.cols + j];
        result.data[j] = s;
      }
      return result;
    } else {
      const result = new Matrix(this.rows, 1);
      for (let i = 0; i < this.rows; i++) {
        let s = 0;
        for (let j = 0; j < this.cols; j++) s += this.data[i * this.cols + j];
        result.data[i] = s;
      }
      return result;
    }
  }

  // Max element
  max() {
    let m = -Infinity;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] > m) m = this.data[i];
    return m;
  }

  // Argmax per row (returns array of column indices)
  argmax() {
    const result = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let maxVal = -Infinity, maxIdx = 0;
      for (let j = 0; j < this.cols; j++) {
        const v = this.data[i * this.cols + j];
        if (v > maxVal) { maxVal = v; maxIdx = j; }
      }
      result[i] = maxIdx;
    }
    return result;
  }

  // Clone
  clone() {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }

  // Pretty print
  toString() {
    let s = `Matrix(${this.rows}×${this.cols}):\n`;
    for (let i = 0; i < Math.min(this.rows, 8); i++) {
      const row = [];
      for (let j = 0; j < Math.min(this.cols, 8); j++) {
        row.push(this.get(i, j).toFixed(4));
      }
      if (this.cols > 8) row.push('...');
      s += '  [' + row.join(', ') + ']\n';
    }
    if (this.rows > 8) s += '  ...\n';
    return s;
  }

  // Static constructors
  // Slice rows [start, end)
  slice(start, end) {
    const rows = end - start;
    const result = new Matrix(rows, this.cols);
    result.data.set(this.data.subarray(start * this.cols, end * this.cols));
    return result;
  }

  // Sum columns → 1×cols vector (alias for sumAxis(0))
  sumRows() { return this.sumAxis(0); }

  static zeros(rows, cols) { return new Matrix(rows, cols); }
  static ones(rows, cols) { return new Matrix(rows, cols).fill(1); }
  static random(rows, cols, scale) { return new Matrix(rows, cols).randomize(scale); }
  static xavier(rows, cols) { return Matrix.random(rows, cols, Math.sqrt(2 / (rows + cols))); }
  static he(rows, cols) { return Matrix.random(rows, cols, Math.sqrt(2 / rows)); }

  static fromArray(arr) {
    if (Array.isArray(arr[0]) || ArrayBuffer.isView(arr[0])) {
      // 2D array (or array of typed arrays)
      const rows = arr.length, cols = arr[0].length;
      const m = new Matrix(rows, cols);
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) m.set(i, j, arr[i][j]);
      return m;
    }
    // 1D array → column vector
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) m.data[i] = arr[i];
    return m;
  }

  // Convert to 2D array
  toArray() {
    const result = [];
    for (let i = 0; i < this.rows; i++) {
      const row = [];
      for (let j = 0; j < this.cols; j++) row.push(this.get(i, j));
      result.push(row);
    }
    return result;
  }

  // One-hot encoding
  static oneHot(indices, numClasses) {
    const m = new Matrix(indices.length, numClasses);
    for (let i = 0; i < indices.length; i++) m.set(i, indices[i], 1);
    return m;
  }
}
