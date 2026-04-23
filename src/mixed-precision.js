// mixed-precision.js — Mixed Precision Training Simulation
// Simulates FP16/BF16 numerical effects for understanding precision tradeoffs.
// In real systems: forward in FP16, backward in FP16, optimizer in FP32.
//
// Key concepts:
// - Loss scaling: multiply loss by large factor before backward to prevent gradient underflow
// - Master weights: keep FP32 copy for optimizer updates
// - Dynamic loss scaling: auto-adjust scale based on overflow detection

/**
 * Simulate FP16 range: min=6.1e-5, max=65504.
 * Rounds to FP16 precision (10-bit mantissa → ~3 decimal digits).
 */
export function toFP16(value) {
  if (!isFinite(value)) return value;
  if (Math.abs(value) > 65504) return value > 0 ? Infinity : -Infinity;
  if (Math.abs(value) < 6.1e-5 && value !== 0) return 0; // Underflow
  
  // Simulate 10-bit mantissa precision
  if (value === 0) return 0;
  const exp = Math.floor(Math.log2(Math.abs(value)));
  const mantissa = Math.abs(value) / Math.pow(2, exp);
  const quantized = Math.round(mantissa * 1024) / 1024;
  return Math.sign(value) * quantized * Math.pow(2, exp);
}

/**
 * Simulate BF16 range: same as FP32 range but 7-bit mantissa (~2 decimal digits).
 */
export function toBF16(value) {
  if (!isFinite(value)) return value;
  if (value === 0) return 0;
  const exp = Math.floor(Math.log2(Math.abs(value)));
  const mantissa = Math.abs(value) / Math.pow(2, exp);
  const quantized = Math.round(mantissa * 128) / 128;
  return Math.sign(value) * quantized * Math.pow(2, exp);
}

/**
 * Dynamic Loss Scaler for mixed precision training.
 */
export class DynamicLossScaler {
  constructor(initScale = 65536, growthFactor = 2, backoffFactor = 0.5, growthInterval = 2000) {
    this.scale = initScale;
    this.growthFactor = growthFactor;
    this.backoffFactor = backoffFactor;
    this.growthInterval = growthInterval;
    this.goodSteps = 0;
    this.overflows = 0;
  }

  /**
   * Scale loss before backward pass.
   */
  scaleUp(loss) {
    return loss * this.scale;
  }

  /**
   * Unscale gradients after backward pass.
   * @param {Float64Array} gradients
   * @returns {{ gradients: Float64Array, hasOverflow: boolean }}
   */
  unscale(gradients) {
    const unscaled = new Float64Array(gradients.length);
    let hasOverflow = false;
    
    for (let i = 0; i < gradients.length; i++) {
      unscaled[i] = gradients[i] / this.scale;
      if (!isFinite(unscaled[i])) hasOverflow = true;
    }
    
    return { gradients: unscaled, hasOverflow };
  }

  /**
   * Update scale based on overflow detection.
   */
  update(hasOverflow) {
    if (hasOverflow) {
      this.scale *= this.backoffFactor;
      this.goodSteps = 0;
      this.overflows++;
    } else {
      this.goodSteps++;
      if (this.goodSteps >= this.growthInterval) {
        this.scale *= this.growthFactor;
        this.goodSteps = 0;
      }
    }
  }

  getStats() {
    return {
      currentScale: this.scale,
      overflows: this.overflows,
      goodSteps: this.goodSteps,
    };
  }
}

/**
 * Simulate full mixed precision forward+backward.
 */
export function mixedPrecisionStep(weights, gradients, lr, scaler) {
  // Cast weights to FP16 for forward/backward
  const fp16Weights = weights.map(toFP16);
  
  // Scale gradients
  const scaledGrads = new Float64Array(gradients.length);
  for (let i = 0; i < gradients.length; i++) scaledGrads[i] = gradients[i] * scaler.scale;
  
  // Cast to FP16
  const fp16Grads = scaledGrads.map(toFP16);
  
  // Unscale
  const { gradients: unscaled, hasOverflow } = scaler.unscale(fp16Grads);
  scaler.update(hasOverflow);
  
  if (hasOverflow) return { weights, skipped: true };
  
  // Update master (FP32) weights
  const updated = new Float64Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    updated[i] = weights[i] - lr * unscaled[i];
  }
  
  return { weights: updated, skipped: false };
}
