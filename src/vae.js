// vae.js — Variational Autoencoder (Kingma & Welling, 2014)
// Learns a latent space representation with smooth, continuous structure.
//
// Encoder: x → μ, log(σ²) — produces mean and log-variance
// Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
// Decoder: z → x̂ — reconstructs the input
// Loss: L = reconstruction_loss + β * KL(q(z|x) || p(z))
//       KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

import { Matrix } from './matrix.js';

function sigmoidScalar(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
function randn() {
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export class VAE {
  /**
   * @param {number} inputSize - Input dimension
   * @param {number} hiddenSize - Hidden layer dimension
   * @param {number} latentSize - Latent space dimension
   * @param {object|number} opts - Options object or beta value
   */
  constructor(inputSize, hiddenSize, latentSize, opts = {}) {
    if (typeof opts === 'number') opts = { beta: opts };
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.latentSize = latentSize;
    this.beta = opts.beta ?? 1.0;
    this.lr = opts.learningRate ?? 0.001;
    
    // Encoder: input → hidden → (mu, logVar)
    this.We1 = Matrix.random(inputSize, hiddenSize).map(v => v * Math.sqrt(2.0 / inputSize));
    this.be1 = new Float64Array(hiddenSize);
    this.Wmu = Matrix.random(hiddenSize, latentSize).map(v => v * Math.sqrt(2.0 / hiddenSize));
    this.bmu = new Float64Array(latentSize);
    this.Wlogvar = Matrix.random(hiddenSize, latentSize).map(v => v * Math.sqrt(2.0 / hiddenSize));
    this.blogvar = new Float64Array(latentSize);
    
    // Decoder: latent → hidden → output
    this.Wd1 = Matrix.random(latentSize, hiddenSize).map(v => v * Math.sqrt(2.0 / latentSize));
    this.bd1 = new Float64Array(hiddenSize);
    this.Wd2 = Matrix.random(hiddenSize, inputSize).map(v => v * Math.sqrt(2.0 / hiddenSize));
    this.bd2 = new Float64Array(inputSize);
  }

  /**
   * Encode input to latent distribution parameters.
   * @param {Matrix|number[]} x - Input (column vector or array)
   * @returns {{ mu: Matrix, logVar: Matrix, hidden: Float64Array }}
   */
  encode(x) {
    const inp = x instanceof Matrix ? Array.from(x.data) : Array.from(x);
    
    // Hidden layer with ReLU
    const hidden = new Float64Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.be1[j];
      for (let i = 0; i < this.inputSize; i++) sum += inp[i] * this.We1.get(i, j);
      hidden[j] = Math.max(0, sum);
    }
    
    // Mean
    const muData = new Float64Array(this.latentSize);
    for (let j = 0; j < this.latentSize; j++) {
      let sum = this.bmu[j];
      for (let i = 0; i < this.hiddenSize; i++) sum += hidden[i] * this.Wmu.get(i, j);
      muData[j] = sum;
    }
    
    // Log-variance
    const logVarData = new Float64Array(this.latentSize);
    for (let j = 0; j < this.latentSize; j++) {
      let sum = this.blogvar[j];
      for (let i = 0; i < this.hiddenSize; i++) sum += hidden[i] * this.Wlogvar.get(i, j);
      logVarData[j] = sum;
    }
    
    const mu = new Matrix(this.latentSize, 1, muData);
    const logVar = new Matrix(this.latentSize, 1, logVarData);
    
    return { mu, logVar, hidden };
  }

  /**
   * Reparameterization trick: z = mu + exp(0.5 * logVar) * epsilon
   */
  reparameterize(mu, logVar) {
    const z = new Matrix(this.latentSize, 1);
    for (let i = 0; i < this.latentSize; i++) {
      const std = Math.exp(0.5 * logVar.data[i]);
      z.data[i] = mu.data[i] + std * randn();
    }
    return z;
  }

  /**
   * Decode from latent space to reconstruction.
   * @param {Matrix} z - Latent vector (latentSize × 1)
   * @returns {Matrix} Reconstruction (inputSize × 1)
   */
  decode(z) {
    const zData = z instanceof Matrix ? z.data : z;
    
    // Hidden layer with ReLU
    const hidden = new Float64Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.bd1[j];
      for (let i = 0; i < this.latentSize; i++) sum += zData[i] * this.Wd1.get(i, j);
      hidden[j] = Math.max(0, sum);
    }
    
    // Output with sigmoid
    const outData = new Float64Array(this.inputSize);
    for (let j = 0; j < this.inputSize; j++) {
      let sum = this.bd2[j];
      for (let i = 0; i < this.hiddenSize; i++) sum += hidden[i] * this.Wd2.get(i, j);
      outData[j] = sigmoidScalar(sum);
    }
    
    return new Matrix(this.inputSize, 1, outData);
  }

  /**
   * Full forward pass.
   * @param {Matrix|number[]} x - Input
   * @returns {{ reconstruction: Matrix, mu: Matrix, logVar: Matrix, z: Matrix }}
   */
  forward(x) {
    const { mu, logVar, hidden } = this.encode(x);
    const z = this.reparameterize(mu, logVar);
    const reconstruction = this.decode(z);
    return { reconstruction, mu, logVar, z, _hidden: hidden, _input: x };
  }

  /**
   * Compute VAE loss.
   * @returns {{ total: number, recon: number, kl: number }}
   */
  computeLoss(input, reconstruction, mu, logVar) {
    const inp = input instanceof Matrix ? input.data : input;
    
    // Binary cross-entropy reconstruction loss
    let recon = 0;
    for (let i = 0; i < this.inputSize; i++) {
      const p = Math.max(1e-8, Math.min(1 - 1e-8, reconstruction.data[i]));
      const t = inp[i];
      recon += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
    }
    
    // KL divergence: -0.5 * sum(1 + logVar - mu^2 - exp(logVar))
    let kl = 0;
    for (let i = 0; i < this.latentSize; i++) {
      kl += -0.5 * (1 + logVar.data[i] - mu.data[i] ** 2 - Math.exp(logVar.data[i]));
    }
    
    const total = recon + this.beta * kl;
    return { total, recon, kl };
  }

  /**
   * Train VAE on data.
   * @param {number[][]} data - Array of input vectors
   * @param {object} opts - { epochs, onEpoch }
   * @returns {{ history: Array<{epoch, loss, recon, kl}> }}
   */
  train(data, opts = {}) {
    const epochs = opts.epochs || 50;
    const history = [];
    
    for (let ep = 0; ep < epochs; ep++) {
      let totalLoss = 0, totalRecon = 0, totalKL = 0;
      
      for (const sample of data) {
        const inp = Array.isArray(sample) ? sample : Array.from(sample);
        
        // Forward
        const { mu, logVar, hidden } = this.encode(inp);
        const z = this.reparameterize(mu, logVar);
        
        // Decode with saved activations
        const decHidden = new Float64Array(this.hiddenSize);
        for (let j = 0; j < this.hiddenSize; j++) {
          let sum = this.bd1[j];
          for (let i = 0; i < this.latentSize; i++) sum += z.data[i] * this.Wd1.get(i, j);
          decHidden[j] = Math.max(0, sum);
        }
        const outPre = new Float64Array(this.inputSize);
        const reconstruction = new Float64Array(this.inputSize);
        for (let j = 0; j < this.inputSize; j++) {
          let sum = this.bd2[j];
          for (let i = 0; i < this.hiddenSize; i++) sum += decHidden[i] * this.Wd2.get(i, j);
          outPre[j] = sum;
          reconstruction[j] = sigmoidScalar(sum);
        }
        
        // Compute loss
        let recon = 0;
        for (let i = 0; i < this.inputSize; i++) {
          const p = Math.max(1e-8, Math.min(1 - 1e-8, reconstruction[i]));
          recon += -(inp[i] * Math.log(p) + (1 - inp[i]) * Math.log(1 - p));
        }
        let kl = 0;
        for (let i = 0; i < this.latentSize; i++) {
          kl += -0.5 * (1 + logVar.data[i] - mu.data[i] ** 2 - Math.exp(logVar.data[i]));
        }
        const loss = recon + this.beta * kl;
        totalLoss += loss;
        totalRecon += recon;
        totalKL += kl;
        
        // === Backpropagation ===
        
        // dL/d(reconstruction) = -(target/p - (1-target)/(1-p))
        // dL/d(outPre) = reconstruction - target (sigmoid derivative cancels)
        const dOutPre = new Float64Array(this.inputSize);
        for (let i = 0; i < this.inputSize; i++) {
          dOutPre[i] = reconstruction[i] - inp[i];
        }
        
        // Gradient for Wd2, bd2
        for (let j = 0; j < this.inputSize; j++) {
          this.bd2[j] -= this.lr * dOutPre[j];
          for (let i = 0; i < this.hiddenSize; i++) {
            this.Wd2.set(i, j, this.Wd2.get(i, j) - this.lr * decHidden[i] * dOutPre[j]);
          }
        }
        
        // dL/d(decHidden)
        const dDecHidden = new Float64Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
          for (let j = 0; j < this.inputSize; j++) {
            dDecHidden[i] += this.Wd2.get(i, j) * dOutPre[j];
          }
          if (decHidden[i] <= 0) dDecHidden[i] = 0; // ReLU
        }
        
        // Gradient for Wd1, bd1
        for (let j = 0; j < this.hiddenSize; j++) {
          this.bd1[j] -= this.lr * dDecHidden[j];
          for (let i = 0; i < this.latentSize; i++) {
            this.Wd1.set(i, j, this.Wd1.get(i, j) - this.lr * z.data[i] * dDecHidden[j]);
          }
        }
        
        // dL/dz
        const dz = new Float64Array(this.latentSize);
        for (let i = 0; i < this.latentSize; i++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            dz[i] += this.Wd1.get(i, j) * dDecHidden[j];
          }
        }
        
        // dL/dmu = dz + mu (KL gradient)
        // dL/dlogvar = 0.5 * dz * exp(0.5*logvar) * epsilon + 0.5*(-1 + exp(logvar)) (KL gradient)
        const dmu = new Float64Array(this.latentSize);
        const dlogvar = new Float64Array(this.latentSize);
        for (let i = 0; i < this.latentSize; i++) {
          dmu[i] = dz[i] + this.beta * mu.data[i];
          const std = Math.exp(0.5 * logVar.data[i]);
          const eps = (z.data[i] - mu.data[i]) / (std + 1e-8);
          dlogvar[i] = dz[i] * 0.5 * std * eps + this.beta * 0.5 * (-1 + Math.exp(logVar.data[i]));
        }
        
        // Gradient for Wmu, bmu
        for (let j = 0; j < this.latentSize; j++) {
          this.bmu[j] -= this.lr * dmu[j];
          for (let i = 0; i < this.hiddenSize; i++) {
            this.Wmu.set(i, j, this.Wmu.get(i, j) - this.lr * hidden[i] * dmu[j]);
          }
        }
        
        // Gradient for Wlogvar, blogvar
        for (let j = 0; j < this.latentSize; j++) {
          this.blogvar[j] -= this.lr * dlogvar[j];
          for (let i = 0; i < this.hiddenSize; i++) {
            this.Wlogvar.set(i, j, this.Wlogvar.get(i, j) - this.lr * hidden[i] * dlogvar[j]);
          }
        }
        
        // Backprop through encoder hidden
        const dEncHidden = new Float64Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
          for (let j = 0; j < this.latentSize; j++) {
            dEncHidden[i] += this.Wmu.get(i, j) * dmu[j] + this.Wlogvar.get(i, j) * dlogvar[j];
          }
          if (hidden[i] <= 0) dEncHidden[i] = 0; // ReLU
        }
        
        // Gradient for We1, be1
        for (let j = 0; j < this.hiddenSize; j++) {
          this.be1[j] -= this.lr * dEncHidden[j];
          for (let i = 0; i < this.inputSize; i++) {
            this.We1.set(i, j, this.We1.get(i, j) - this.lr * inp[i] * dEncHidden[j]);
          }
        }
      }
      
      const avgLoss = totalLoss / data.length;
      const epochData = { epoch: ep, loss: avgLoss, recon: totalRecon / data.length, kl: totalKL / data.length };
      history.push(epochData);
      if (opts.onEpoch) opts.onEpoch(epochData);
    }
    
    return { history };
  }

  /**
   * Generate samples from the prior p(z) = N(0, I).
   * @param {number} n - Number of samples
   * @returns {Matrix[]} Array of generated samples
   */
  generate(n) {
    const samples = [];
    for (let i = 0; i < n; i++) {
      const z = new Matrix(this.latentSize, 1);
      for (let j = 0; j < this.latentSize; j++) z.data[j] = randn();
      samples.push(this.decode(z));
    }
    return samples;
  }

  /**
   * Interpolate between two inputs in latent space.
   * @param {number[]} a - First input
   * @param {number[]} b - Second input
   * @param {number} steps - Number of interpolation steps
   * @returns {Matrix[]} Array of steps+1 reconstructions
   */
  interpolate(a, b, steps) {
    const { mu: muA } = this.encode(a);
    const { mu: muB } = this.encode(b);
    
    const results = [];
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const z = new Matrix(this.latentSize, 1);
      for (let i = 0; i < this.latentSize; i++) {
        z.data[i] = (1 - t) * muA.data[i] + t * muB.data[i];
      }
      results.push(this.decode(z));
    }
    return results;
  }

  toJSON() {
    return {
      type: 'VAE',
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      latentSize: this.latentSize,
      beta: this.beta,
      We1: Array.from(this.We1.data),
      be1: Array.from(this.be1),
      Wmu: Array.from(this.Wmu.data),
      bmu: Array.from(this.bmu),
      Wlogvar: Array.from(this.Wlogvar.data),
      blogvar: Array.from(this.blogvar),
      Wd1: Array.from(this.Wd1.data),
      bd1: Array.from(this.bd1),
      Wd2: Array.from(this.Wd2.data),
      bd2: Array.from(this.bd2),
    };
  }
}
