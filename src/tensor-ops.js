// tensor-ops.js — Basic tensor operations (no external deps)
export function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function outerProduct(a, b) {
  const result = [];
  for (let i = 0; i < a.length; i++) {
    result.push(b.map(v => a[i] * v));
  }
  return result;
}

export function hadamard(a, b) {
  return a.map((v, i) => v * b[i]);
}

export function l2Norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

export function normalize(v) {
  const norm = l2Norm(v) + 1e-8;
  return v.map(x => x / norm);
}
