// capsule.js — Capsule Networks
// Dynamic routing, squash activation, capsule layers
// Based on "Dynamic Routing Between Capsules" (Sabour, Frosst, Hinton, 2017)

// ===== Squash Activation =====
// Non-linear activation for capsule vectors
// Preserves direction, scales length to [0, 1]
// squash(s) = ||s||² / (1 + ||s||²) * s / ||s||
export function squash(vector) {
  const normSq = vector.reduce((s, v) => s + v * v, 0);
  const norm = Math.sqrt(normSq + 1e-8);
  const scale = normSq / (1 + normSq) / norm;
  return vector.map(v => v * scale);
}

// Gradient of squash(s) w.r.t. s, applied to upstream gradient dv
// Returns ds = J_squash^T · dv
export function squashBackward(s, dv) {
  const normSq = s.reduce((sum, v) => sum + v * v, 0);
  const norm = Math.sqrt(normSq + 1e-8);
  const normCubed = norm * norm * norm;
  const denom = 1 + normSq;
  const denomSq = denom * denom;
  
  // d(squash)/ds_k = (normSq/(denom*norm)) * (delta_ij - s_i*s_j/normSq) + (2*s_k*s_i)/(denom^2*norm) - (normSq*s_k*s_i)/(denom*normCubed)
  // Simplified: use the Jacobian-vector product directly
  // Let f = normSq / (denom * norm), then squash = f * s
  // df/ds_k = 2*s_k / (denom*norm) - normSq*s_k / (denom*normCubed) - 2*normSq*s_k / (denomSq*norm)
  
  const f = normSq / (denom * norm);
  const ds = new Array(s.length);
  
  // df_k = ds_k factor from chain rule
  // d(f*s_i)/ds_k = f*delta_ik + s_i * df/ds_k
  // So (J*dv)_k = f*dv_k + sum_i(s_i * dv_i) * df/ds_k
  
  const sdotdv = s.reduce((sum, si, i) => sum + si * dv[i], 0);
  
  for (let k = 0; k < s.length; k++) {
    const dfk = 2 * s[k] / (denom * norm) 
              - normSq * s[k] / (denom * normCubed) 
              - 2 * normSq * s[k] / (denomSq * norm);
    ds[k] = f * dv[k] + sdotdv * dfk;
  }
  
  return ds;
}

// Vector length (capsule "probability")
export function vectorNorm(vector) {
  return Math.sqrt(vector.reduce((s, v) => s + v * v, 0));
}

// ===== Capsule Layer =====
export class CapsuleLayer {
  constructor(numCapsules, capsuleDim, inputCapsules, inputDim, routingIterations = 3) {
    this.numCapsules = numCapsules;
    this.capsuleDim = capsuleDim;
    this.inputCapsules = inputCapsules;
    this.inputDim = inputDim;
    this.routingIterations = routingIterations;

    // Transformation matrices: W[i][j] transforms input capsule i for output capsule j
    // Shape: [inputCapsules][numCapsules] of [capsuleDim x inputDim]
    this.W = [];
    for (let i = 0; i < inputCapsules; i++) {
      this.W[i] = [];
      for (let j = 0; j < numCapsules; j++) {
        // Xavier-like initialization
        const scale = Math.sqrt(2 / (inputDim + capsuleDim));
        this.W[i][j] = Array.from({ length: capsuleDim }, () =>
          Array.from({ length: inputDim }, () => (Math.random() * 2 - 1) * scale)
        );
      }
    }

    // Cache
    this.inputCaps = null;
    this.predictions = null; // u_hat[i][j] = W[i][j] * input[i]
    this.couplingCoeffs = null; // c[i][j]
    this.output = null;
  }

  // Transform matrix-vector multiply
  matVecMul(mat, vec) {
    return mat.map(row => row.reduce((s, w, k) => s + w * vec[k], 0));
  }

  forward(inputs) {
    // inputs: array of [inputCapsules] arrays of [inputDim] vectors
    this.inputCaps = inputs;

    // Compute prediction vectors: u_hat[i][j] = W[i][j] * u[i]
    this.predictions = [];
    for (let i = 0; i < this.inputCapsules; i++) {
      this.predictions[i] = [];
      for (let j = 0; j < this.numCapsules; j++) {
        this.predictions[i][j] = this.matVecMul(this.W[i][j], inputs[i]);
      }
    }

    // Dynamic routing
    // Initialize routing logits b[i][j] = 0
    const b = [];
    for (let i = 0; i < this.inputCapsules; i++) {
      b[i] = new Array(this.numCapsules).fill(0);
    }

    let output = null;

    for (let iter = 0; iter < this.routingIterations; iter++) {
      // Softmax over j for each i: c[i][j] = exp(b[i][j]) / sum_k exp(b[i][k])
      this.couplingCoeffs = [];
      for (let i = 0; i < this.inputCapsules; i++) {
        const maxB = Math.max(...b[i]);
        const exps = b[i].map(v => Math.exp(v - maxB));
        const sumExp = exps.reduce((a, v) => a + v, 0);
        this.couplingCoeffs[i] = exps.map(e => e / sumExp);
      }

      // Compute weighted sum: s[j] = sum_i c[i][j] * u_hat[i][j]
      const s = [];
      for (let j = 0; j < this.numCapsules; j++) {
        s[j] = new Array(this.capsuleDim).fill(0);
        for (let i = 0; i < this.inputCapsules; i++) {
          const c = this.couplingCoeffs[i][j];
          for (let d = 0; d < this.capsuleDim; d++) {
            s[j][d] += c * this.predictions[i][j][d];
          }
        }
      }

      // Squash: v[j] = squash(s[j])
      this._preSquash = s; // Cache for backward
      output = s.map(sj => squash(sj));

      // Update routing logits (except last iteration)
      if (iter < this.routingIterations - 1) {
        for (let i = 0; i < this.inputCapsules; i++) {
          for (let j = 0; j < this.numCapsules; j++) {
            // Agreement: u_hat[i][j] · v[j]
            let agreement = 0;
            for (let d = 0; d < this.capsuleDim; d++) {
              agreement += this.predictions[i][j][d] * output[j][d];
            }
            b[i][j] += agreement;
          }
        }
      }
    }

    this.output = output;
    return output;
  }

  // Backward pass
  backward(dOutput) {
    // dOutput: [numCapsules][capsuleDim] gradients

    // Gradient through squash using proper Jacobian
    const dS = dOutput.map((dv, j) => {
      return squashBackward(this._preSquash[j], dv);
    });

    // Gradient for transformation weights
    const dW = [];
    for (let i = 0; i < this.inputCapsules; i++) {
      dW[i] = [];
      for (let j = 0; j < this.numCapsules; j++) {
        const c = this.couplingCoeffs[i][j];
        dW[i][j] = this.W[i][j].map((row, d) =>
          row.map((_, k) => c * dS[j][d] * this.inputCaps[i][k])
        );
      }
    }

    // Update weights
    const lr = 0.01;
    for (let i = 0; i < this.inputCapsules; i++) {
      for (let j = 0; j < this.numCapsules; j++) {
        for (let d = 0; d < this.capsuleDim; d++) {
          for (let k = 0; k < this.inputDim; k++) {
            this.W[i][j][d][k] -= lr * dW[i][j][d][k];
          }
        }
      }
    }

    // Input gradients
    const dInput = [];
    for (let i = 0; i < this.inputCapsules; i++) {
      dInput[i] = new Array(this.inputDim).fill(0);
      for (let j = 0; j < this.numCapsules; j++) {
        const c = this.couplingCoeffs[i][j];
        for (let k = 0; k < this.inputDim; k++) {
          for (let d = 0; d < this.capsuleDim; d++) {
            dInput[i][k] += c * dS[j][d] * this.W[i][j][d][k];
          }
        }
      }
    }

    return dInput;
  }

  paramCount() {
    return this.inputCapsules * this.numCapsules * this.capsuleDim * this.inputDim;
  }
}

// ===== Margin Loss (for CapsNet classification) =====
// L_k = T_k * max(0, m+ - ||v_k||)² + λ * (1 - T_k) * max(0, ||v_k|| - m-)²
export function marginLoss(capsuleOutputs, labels, { mPlus = 0.9, mMinus = 0.1, lambda = 0.5 } = {}) {
  let totalLoss = 0;
  const numClasses = capsuleOutputs.length;
  const dOutput = [];

  for (let k = 0; k < numClasses; k++) {
    const vNorm = vectorNorm(capsuleOutputs[k]);
    const tk = labels[k] || 0;

    // Positive loss (present class)
    const posMargin = Math.max(0, mPlus - vNorm);
    const posLoss = tk * posMargin * posMargin;

    // Negative loss (absent class)
    const negMargin = Math.max(0, vNorm - mMinus);
    const negLoss = lambda * (1 - tk) * negMargin * negMargin;

    totalLoss += posLoss + negLoss;

    // Gradient w.r.t. capsule output
    const scale = vNorm > 1e-8 ? 1 / vNorm : 0;
    dOutput[k] = capsuleOutputs[k].map(v => {
      let grad = 0;
      if (tk > 0 && posMargin > 0) {
        grad -= 2 * tk * posMargin * scale * v;
      }
      if ((1 - tk) > 0 && negMargin > 0) {
        grad += 2 * lambda * (1 - tk) * negMargin * scale * v;
      }
      return grad;
    });
  }

  return { loss: totalLoss / numClasses, gradients: dOutput };
}

// ===== Primary Capsules =====
// Convert flat feature vector into capsule format
export function primaryCapsules(features, numCapsules, capsuleDim) {
  const capsules = [];
  let idx = 0;
  for (let i = 0; i < numCapsules; i++) {
    const cap = [];
    for (let d = 0; d < capsuleDim; d++) {
      cap.push(features[idx % features.length] || 0);
      idx++;
    }
    capsules.push(squash(cap));
  }
  return capsules;
}
