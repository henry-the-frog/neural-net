// rope-scaling.js — RoPE scaling methods for extending context length
// NTK-aware scaling (Reddit), YaRN, Linear interpolation

export function linearRoPEScaling(freq, scaleFactor) {
  return freq / scaleFactor;
}

export function ntkAwareScaling(freq, dim, scaleFactor) {
  // NTK-aware: scale the base frequency instead of dividing positions
  const newBase = 10000 * Math.pow(scaleFactor, dim / (dim - 2));
  return 1 / Math.pow(newBase, 2 * Math.floor(dim / 2) / dim);
}

export function yarnScaling(freq, dim, scaleFactor, alpha = 1, beta = 32) {
  // YaRN: different scaling for different frequency bands
  const wavelength = 2 * Math.PI / freq;
  if (wavelength < beta) return freq; // High-freq: no scaling
  if (wavelength > alpha * scaleFactor) return freq / scaleFactor; // Low-freq: full scaling
  // Mid-freq: interpolate
  const t = (wavelength - beta) / (alpha * scaleFactor - beta);
  return freq / (1 + (scaleFactor - 1) * t);
}
