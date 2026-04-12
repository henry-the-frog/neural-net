// diff-sort.js — Differentiable Sorting
// Continuous relaxation of sorting for gradient-based optimization
// Based on optimal transport / Sinkhorn operations

// ===== Soft Sort (via optimal transport) =====
// Approximate permutation matrix using Sinkhorn iterations
export function softSort(values, temperature = 1, iterations = 20) {
  const N = values.length;

  // Pairwise distance matrix between values and sorted values
  const sorted = [...values].sort((a, b) => a - b);

  // Cost matrix: C[i][j] = -|value_i - sorted_j|²
  const C = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) =>
      -((values[i] - sorted[j]) ** 2) / temperature
    )
  );

  // Sinkhorn normalization to get doubly-stochastic matrix
  const P = sinkhorn(C, iterations);

  // Soft-sorted output: P * sorted
  const output = Array.from({ length: N }, (_, i) => {
    let sum = 0;
    for (let j = 0; j < N; j++) sum += P[i][j] * sorted[j];
    return sum;
  });

  return { output, permutation: P, sorted };
}

// ===== Sinkhorn Algorithm =====
// Normalize matrix to be doubly stochastic (rows and columns sum to 1)
export function sinkhorn(logMatrix, iterations = 20) {
  const N = logMatrix.length;

  // Exponentiate
  let M = logMatrix.map(row => {
    const max = Math.max(...row);
    return row.map(v => Math.exp(v - max));
  });

  for (let iter = 0; iter < iterations; iter++) {
    // Row normalization
    M = M.map(row => {
      const sum = row.reduce((a, b) => a + b, 0) + 1e-10;
      return row.map(v => v / sum);
    });

    // Column normalization
    for (let j = 0; j < N; j++) {
      let colSum = 0;
      for (let i = 0; i < N; i++) colSum += M[i][j];
      colSum += 1e-10;
      for (let i = 0; i < N; i++) M[i][j] /= colSum;
    }
  }

  return M;
}

// ===== Soft Rank =====
// Get soft ranks (differentiable version of argsort)
export function softRank(values, temperature = 1) {
  const N = values.length;
  const ranks = Array.from({ length: N }, (_, i) => {
    let rank = 0;
    for (let j = 0; j < N; j++) {
      if (i !== j) {
        // Sigmoid approximation of indicator function
        rank += 1 / (1 + Math.exp(-(values[j] - values[i]) / temperature));
      }
    }
    return rank;
  });
  return ranks;
}

// ===== Differentiable Top-K =====
// Soft selection of top-K elements
export function softTopK(values, k, temperature = 1) {
  const ranks = softRank(values, temperature);
  // Soft indicator: is this in top-k?
  const indicators = ranks.map(r =>
    1 / (1 + Math.exp((r - k + 0.5) / temperature))
  );
  // Weighted values
  const selected = values.map((v, i) => v * indicators[i]);
  return { selected, indicators, ranks };
}

// ===== NeuralSort =====
// From "Grover et al., Stochastic Optimization of Sorting Networks"
export function neuralSort(scores, temperature = 1) {
  const N = scores.length;
  // For each permutation position, compute soft assignment
  const P = Array.from({ length: N }, (_, i) => {
    const row = scores.map((s, j) => {
      // How likely is element j to be in position i?
      let logit = 0;
      for (let k = 0; k < N; k++) {
        if (k !== j) {
          logit += Math.log(1 + Math.exp(-(scores[k] - scores[j]) / temperature));
        }
      }
      return -(N - 1 - i) * scores[j] / temperature - logit;
    });

    // Softmax
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  });

  return P;
}

// ===== Utility =====
export function isDoublyStochastic(matrix, tol = 0.05) {
  const N = matrix.length;
  // Check rows sum to 1
  for (let i = 0; i < N; i++) {
    const rowSum = matrix[i].reduce((a, b) => a + b, 0);
    if (Math.abs(rowSum - 1) > tol) return false;
  }
  // Check columns sum to 1
  for (let j = 0; j < N; j++) {
    let colSum = 0;
    for (let i = 0; i < N; i++) colSum += matrix[i][j];
    if (Math.abs(colSum - 1) > tol) return false;
  }
  return true;
}
