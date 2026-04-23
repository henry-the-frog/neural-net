// weight-init.js — Weight Initialization Strategies
export function xavierUniform(fanIn, fanOut) {
  const limit = Math.sqrt(6.0 / (fanIn + fanOut));
  return () => (Math.random() * 2 - 1) * limit;
}

export function xavierNormal(fanIn, fanOut) {
  const std = Math.sqrt(2.0 / (fanIn + fanOut));
  return () => randn() * std;
}

export function kaimingUniform(fanIn) {
  const limit = Math.sqrt(6.0 / fanIn);
  return () => (Math.random() * 2 - 1) * limit;
}

export function kaimingNormal(fanIn) {
  const std = Math.sqrt(2.0 / fanIn);
  return () => randn() * std;
}

export function orthogonal(n) {
  // Simplified: generate random matrix and orthogonalize via Gram-Schmidt
  const M = Array.from({length: n}, () => Array.from({length: n}, () => randn()));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < n; k++) dot += M[i][k] * M[j][k];
      for (let k = 0; k < n; k++) M[i][k] -= dot * M[j][k];
    }
    let norm = 0;
    for (let k = 0; k < n; k++) norm += M[i][k] * M[i][k];
    norm = Math.sqrt(norm);
    for (let k = 0; k < n; k++) M[i][k] /= norm;
  }
  return M;
}

function randn() {
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
