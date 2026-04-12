// normalizing-flows.js — Invertible Transforms for Generative Modeling
// Transform simple distributions (Gaussian) into complex ones via bijective maps

// ===== Planar Flow =====
// f(z) = z + u * tanh(w^T z + b)
// Simple but expressive invertible transform
export class PlanarFlow {
  constructor(dim) {
    this.dim = dim;
    this.w = Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.5);
    this.u = Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.5);
    this.b = (Math.random() - 0.5) * 0.1;

    // Ensure invertibility: u' = u + (m(w^T u) - w^T u) * w / ||w||²
    this._ensureInvertible();
  }

  _ensureInvertible() {
    const wtu = dot(this.w, this.u);
    const m = -1 + Math.log(1 + Math.exp(wtu)); // softplus
    if (m !== wtu) {
      const wnorm2 = dot(this.w, this.w);
      if (wnorm2 > 0) {
        const correction = (m - wtu) / wnorm2;
        this.u = this.u.map((u, i) => u + correction * this.w[i]);
      }
    }
  }

  forward(z) {
    const wtz = dot(this.w, z) + this.b;
    const h = Math.tanh(wtz);
    const output = z.map((zi, i) => zi + this.u[i] * h);

    // Log-determinant of Jacobian
    const hPrime = 1 - h * h; // dtanh/dx
    const psi = this.w.map(wi => hPrime * wi);
    const det = 1 + dot(this.u, psi);
    const logDetJ = Math.log(Math.abs(det) + 1e-8);

    return { z: output, logDetJ };
  }

  // Approximate inverse (iterative)
  inverse(y, iterations = 20) {
    let z = [...y];
    for (let iter = 0; iter < iterations; iter++) {
      const wtz = dot(this.w, z) + this.b;
      const h = Math.tanh(wtz);
      z = y.map((yi, i) => yi - this.u[i] * h);
    }
    return z;
  }

  paramCount() { return this.dim * 2 + 1; }
}

// ===== Affine Coupling Layer =====
// Split input: x1, x2
// y1 = x1 (unchanged)
// y2 = x2 * exp(s(x1)) + t(x1)
// Perfectly invertible with tractable Jacobian
export class AffineCouplingLayer {
  constructor(dim, splitIdx = null) {
    this.dim = dim;
    this.splitIdx = splitIdx || Math.floor(dim / 2);
    const dim1 = this.splitIdx;
    const dim2 = dim - dim1;

    // Scale network: x1 → s (log-scale for x2)
    // Simple: linear transform + tanh to bound scale
    this.sWeights = Array.from({ length: dim2 }, () =>
      Array.from({ length: dim1 }, () => (Math.random() - 0.5) * 0.3)
    );
    this.sBias = new Array(dim2).fill(0);

    // Translation network: x1 → t (shift for x2)
    this.tWeights = Array.from({ length: dim2 }, () =>
      Array.from({ length: dim1 }, () => (Math.random() - 0.5) * 0.3)
    );
    this.tBias = new Array(dim2).fill(0);
  }

  _computeST(x1) {
    const dim2 = this.dim - this.splitIdx;
    const s = new Array(dim2);
    const t = new Array(dim2);
    for (let j = 0; j < dim2; j++) {
      let sSum = this.sBias[j], tSum = this.tBias[j];
      for (let i = 0; i < this.splitIdx; i++) {
        sSum += this.sWeights[j][i] * x1[i];
        tSum += this.tWeights[j][i] * x1[i];
      }
      s[j] = Math.tanh(sSum) * 2; // Bound scale
      t[j] = tSum;
    }
    return { s, t };
  }

  forward(x) {
    const x1 = x.slice(0, this.splitIdx);
    const x2 = x.slice(this.splitIdx);
    const { s, t } = this._computeST(x1);

    const y2 = x2.map((v, i) => v * Math.exp(s[i]) + t[i]);
    const logDetJ = s.reduce((a, b) => a + b, 0); // sum of log-scales

    return { z: [...x1, ...y2], logDetJ };
  }

  inverse(y) {
    const y1 = y.slice(0, this.splitIdx);
    const y2 = y.slice(this.splitIdx);
    const { s, t } = this._computeST(y1);

    const x2 = y2.map((v, i) => (v - t[i]) * Math.exp(-s[i]));
    return [...y1, ...x2];
  }

  paramCount() {
    const dim2 = this.dim - this.splitIdx;
    return dim2 * this.splitIdx * 2 + dim2 * 2;
  }
}

// ===== ActNorm (Activation Normalization) =====
// Learnable per-channel scale and bias, initialized from data
export class ActNorm {
  constructor(dim) {
    this.dim = dim;
    this.scale = new Array(dim).fill(1);
    this.bias = new Array(dim).fill(0);
    this.initialized = false;
  }

  // Data-dependent initialization
  initialize(batch) {
    const mean = new Array(this.dim).fill(0);
    const variance = new Array(this.dim).fill(0);
    const N = batch.length;

    for (const x of batch) {
      for (let d = 0; d < this.dim; d++) mean[d] += x[d];
    }
    for (let d = 0; d < this.dim; d++) mean[d] /= N;

    for (const x of batch) {
      for (let d = 0; d < this.dim; d++) variance[d] += (x[d] - mean[d]) ** 2;
    }

    this.bias = mean.map(m => -m);
    this.scale = variance.map(v => 1 / Math.sqrt(v / N + 1e-6));
    this.initialized = true;
  }

  forward(x) {
    const z = x.map((v, i) => (v + this.bias[i]) * this.scale[i]);
    const logDetJ = this.scale.reduce((s, v) => s + Math.log(Math.abs(v) + 1e-8), 0);
    return { z, logDetJ };
  }

  inverse(z) {
    return z.map((v, i) => v / this.scale[i] - this.bias[i]);
  }
}

// ===== Normalizing Flow (stack of layers) =====
export class NormalizingFlow {
  constructor(layers) {
    this.layers = layers;
  }

  // Transform from data space to latent space
  forward(x) {
    let z = [...x];
    let totalLogDetJ = 0;

    for (const layer of this.layers) {
      const result = layer.forward(z);
      z = result.z;
      totalLogDetJ += result.logDetJ;
    }

    return { z, logDetJ: totalLogDetJ };
  }

  // Transform from latent space to data space (inverse)
  inverse(z) {
    let x = [...z];
    for (let i = this.layers.length - 1; i >= 0; i--) {
      x = this.layers[i].inverse(x);
    }
    return x;
  }

  // Log-likelihood under standard Gaussian prior
  logLikelihood(x) {
    const { z, logDetJ } = this.forward(x);
    // Log p(z) under standard Gaussian
    const logPz = z.reduce((s, zi) => s - 0.5 * (zi * zi + Math.log(2 * Math.PI)), 0);
    return logPz + logDetJ;
  }

  // Sample from the learned distribution
  sample() {
    // Sample from standard Gaussian
    const z = this.layers[0] ?
      Array.from({ length: this.layers[0].dim || 2 }, () => gaussianRandom()) :
      [gaussianRandom(), gaussianRandom()];
    return this.inverse(z);
  }

  paramCount() {
    return this.layers.reduce((s, l) => s + l.paramCount(), 0);
  }
}

// ===== Utility =====
function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

function gaussianRandom() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}
