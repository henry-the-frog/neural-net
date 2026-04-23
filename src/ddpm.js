// ddpm.js — Denoising Diffusion Probabilistic Models (Ho et al., 2020)
// Forward process: gradually add noise to data
// Reverse process: learn to denoise step by step
//
// Forward: q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t) * I)
// Reverse: p_θ(x_{t-1} | x_t) — learned denoiser
//
// Training: minimize || ε - ε_θ(x_t, t) ||² (predict noise)

/**
 * DDPM Noise Scheduler.
 * Manages the beta schedule and derived quantities.
 */
export class DDPMScheduler {
  /**
   * @param {number} numTimesteps - Total diffusion steps (e.g., 1000)
   * @param {number} betaStart - Starting noise level
   * @param {number} betaEnd - Ending noise level
   * @param {string} schedule - 'linear' or 'cosine'
   */
  constructor(numTimesteps = 1000, betaStart = 0.0001, betaEnd = 0.02, schedule = 'linear') {
    this.T = numTimesteps;
    
    // Beta schedule
    this.betas = new Float64Array(numTimesteps);
    if (schedule === 'linear') {
      for (let t = 0; t < numTimesteps; t++) {
        this.betas[t] = betaStart + (betaEnd - betaStart) * t / (numTimesteps - 1);
      }
    } else if (schedule === 'cosine') {
      // Cosine schedule (Nichol & Dhariwal, 2021)
      const alphaBarFn = (t) => Math.cos((t / numTimesteps + 0.008) / 1.008 * Math.PI / 2) ** 2;
      for (let t = 0; t < numTimesteps; t++) {
        this.betas[t] = Math.min(1 - alphaBarFn(t + 1) / alphaBarFn(t), 0.999);
      }
    }
    
    // Derived quantities
    this.alphas = new Float64Array(numTimesteps);
    this.alphasCumprod = new Float64Array(numTimesteps); // ᾱ_t
    this.sqrtAlphasCumprod = new Float64Array(numTimesteps);
    this.sqrtOneMinusAlphasCumprod = new Float64Array(numTimesteps);
    this.posteriorVariance = new Float64Array(numTimesteps);
    
    let cumprod = 1;
    for (let t = 0; t < numTimesteps; t++) {
      this.alphas[t] = 1 - this.betas[t];
      cumprod *= this.alphas[t];
      this.alphasCumprod[t] = cumprod;
      this.sqrtAlphasCumprod[t] = Math.sqrt(cumprod);
      this.sqrtOneMinusAlphasCumprod[t] = Math.sqrt(1 - cumprod);
      
      // Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
      const alphaBarPrev = t > 0 ? this.alphasCumprod[t - 1] : 1;
      this.posteriorVariance[t] = this.betas[t] * (1 - alphaBarPrev) / (1 - this.alphasCumprod[t] + 1e-20);
    }
  }

  /**
   * Forward diffusion: add noise to x_0 to get x_t.
   * q(x_t | x_0) = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
   * @param {Float64Array} x0 - Clean data
   * @param {number} t - Timestep
   * @param {Float64Array} noise - Standard Gaussian noise (optional)
   * @returns {{ xt: Float64Array, noise: Float64Array }}
   */
  addNoise(x0, t, noise = null) {
    const n = x0.length;
    if (!noise) {
      noise = new Float64Array(n);
      for (let i = 0; i < n; i++) noise[i] = randn();
    }
    
    const sqrtAlphaBar = this.sqrtAlphasCumprod[t];
    const sqrtOneMinusAlphaBar = this.sqrtOneMinusAlphasCumprod[t];
    
    const xt = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      xt[i] = sqrtAlphaBar * x0[i] + sqrtOneMinusAlphaBar * noise[i];
    }
    
    return { xt, noise };
  }

  /**
   * Reverse denoising step: predict x_{t-1} from x_t.
   * @param {Float64Array} xt - Noisy data at step t
   * @param {Float64Array} predictedNoise - Model's noise prediction ε_θ
   * @param {number} t - Current timestep
   * @returns {Float64Array} x_{t-1}
   */
  denoise(xt, predictedNoise, t) {
    const n = xt.length;
    const alpha = this.alphas[t];
    const alphaBar = this.alphasCumprod[t];
    const beta = this.betas[t];
    
    // Mean: μ_θ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
    const coeff = beta / this.sqrtOneMinusAlphasCumprod[t];
    const sqrtAlpha = Math.sqrt(alpha);
    
    const mean = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      mean[i] = (xt[i] - coeff * predictedNoise[i]) / sqrtAlpha;
    }
    
    // Add noise (except at t=0)
    if (t > 0) {
      const sigma = Math.sqrt(this.posteriorVariance[t]);
      for (let i = 0; i < n; i++) {
        mean[i] += sigma * randn();
      }
    }
    
    return mean;
  }

  /**
   * Compute training loss (simplified MSE on noise prediction).
   * @param {Float64Array} noise - True noise
   * @param {Float64Array} predictedNoise - Predicted noise
   * @returns {number} MSE loss
   */
  loss(noise, predictedNoise) {
    let mse = 0;
    for (let i = 0; i < noise.length; i++) {
      mse += (noise[i] - predictedNoise[i]) ** 2;
    }
    return mse / noise.length;
  }

  /**
   * Sample random timestep uniformly.
   */
  sampleTimestep() {
    return Math.floor(Math.random() * this.T);
  }
}

// Box-Muller transform for Gaussian random numbers
function randn() {
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export { randn };
