// mdn.js — Mixture Density Networks
// Output a mixture of Gaussians instead of point predictions
// Great for multi-modal distributions (e.g., inverse kinematics)

// ===== Gaussian Mixture =====
export function gaussianPDF(x, mean, variance) {
  return Math.exp(-0.5 * (x - mean) ** 2 / variance) / Math.sqrt(2 * Math.PI * variance);
}

export function logGaussianPDF(x, mean, variance) {
  return -0.5 * (Math.log(2 * Math.PI * variance) + (x - mean) ** 2 / variance);
}

// ===== MDN Output Parsing =====
// MDN output: [pi_1..pi_K, mu_1..mu_K, sigma_1..sigma_K] for K components
export function parseMDNOutput(output, numComponents) {
  const K = numComponents;
  const piLogits = output.slice(0, K);
  const mus = output.slice(K, 2 * K);
  const sigmaRaw = output.slice(2 * K, 3 * K);

  // Softmax for mixing coefficients
  const maxPi = Math.max(...piLogits);
  const piExps = piLogits.map(l => Math.exp(l - maxPi));
  const piSum = piExps.reduce((a, b) => a + b, 0);
  const pi = piExps.map(e => e / piSum);

  // Ensure positive variance via exp
  const sigma = sigmaRaw.map(s => Math.exp(s) + 1e-6);

  return { pi, mu: mus, sigma };
}

// ===== MDN Loss (Negative Log-Likelihood) =====
export function mdnLoss(output, target, numComponents) {
  const { pi, mu, sigma } = parseMDNOutput(output, numComponents);

  // p(target) = sum_k pi_k * N(target | mu_k, sigma_k)
  let logLikelihood = 0;
  const componentProbs = pi.map((p, k) =>
    p * gaussianPDF(target, mu[k], sigma[k] ** 2)
  );
  const totalProb = componentProbs.reduce((a, b) => a + b, 0);
  logLikelihood = -Math.log(totalProb + 1e-10);

  return logLikelihood;
}

// ===== Sampling from MDN =====
export function sampleMDN(output, numComponents) {
  const { pi, mu, sigma } = parseMDNOutput(output, numComponents);

  // Sample component
  let r = Math.random();
  let component = 0;
  for (let k = 0; k < pi.length; k++) {
    r -= pi[k];
    if (r <= 0) { component = k; break; }
  }

  // Sample from chosen Gaussian
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mu[component] + sigma[component] * z;
}

// ===== Multi-dimensional MDN =====
export function parseMDNOutputMultiDim(output, numComponents, outputDim) {
  const K = numComponents;
  const D = outputDim;

  const piLogits = output.slice(0, K);
  const muFlat = output.slice(K, K + K * D);
  const sigmaRaw = output.slice(K + K * D, K + K * D + K * D);

  // Softmax mixing coefficients
  const maxPi = Math.max(...piLogits);
  const piExps = piLogits.map(l => Math.exp(l - maxPi));
  const piSum = piExps.reduce((a, b) => a + b, 0);
  const pi = piExps.map(e => e / piSum);

  // Parse means and sigmas per component
  const mu = [];
  const sigma = [];
  for (let k = 0; k < K; k++) {
    mu.push(muFlat.slice(k * D, (k + 1) * D));
    sigma.push(sigmaRaw.slice(k * D, (k + 1) * D).map(s => Math.exp(s) + 1e-6));
  }

  return { pi, mu, sigma };
}

// ===== Required MDN output size =====
export function mdnOutputSize(numComponents, outputDim = 1) {
  return numComponents * (1 + 2 * outputDim); // pi + mu + sigma per component per dim
}
