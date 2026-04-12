// scheduler.js — Learning rate schedulers
// Control how the learning rate changes during training

/**
 * Step decay: multiply LR by factor every N epochs
 * @param {number} initialLR - Starting learning rate
 * @param {number} factor - Decay factor (0-1)
 * @param {number} stepSize - Epochs between decays
 */
export function stepDecay(initialLR, factor = 0.5, stepSize = 10) {
  return (epoch) => initialLR * Math.pow(factor, Math.floor(epoch / stepSize));
}

/**
 * Exponential decay: LR = initialLR * decay^epoch
 * @param {number} initialLR
 * @param {number} decay - Decay rate per epoch (e.g., 0.99)
 */
export function exponentialDecay(initialLR, decay = 0.99) {
  return (epoch) => initialLR * Math.pow(decay, epoch);
}

/**
 * Cosine annealing: smooth cosine decay to minimum LR
 * @param {number} initialLR
 * @param {number} totalEpochs - Total training epochs
 * @param {number} minLR - Minimum learning rate
 */
export function cosineAnnealing(initialLR, totalEpochs, minLR = 0) {
  return (epoch) => {
    const progress = Math.min(epoch / totalEpochs, 1);
    return minLR + (initialLR - minLR) * 0.5 * (1 + Math.cos(Math.PI * progress));
  };
}

/**
 * Warmup + decay: linear warmup for N epochs, then constant or decay
 * @param {number} targetLR - Target learning rate after warmup
 * @param {number} warmupEpochs - Number of warmup epochs
 * @param {function} afterWarmup - Optional scheduler for after warmup
 */
export function warmup(targetLR, warmupEpochs, afterWarmup = null) {
  return (epoch) => {
    if (epoch < warmupEpochs) {
      // Linear warmup: 0 → targetLR
      return targetLR * (epoch + 1) / warmupEpochs;
    }
    if (afterWarmup) {
      return afterWarmup(epoch - warmupEpochs);
    }
    return targetLR;
  };
}

/**
 * Warmup + cosine annealing: popular modern schedule
 * @param {number} initialLR
 * @param {number} warmupEpochs
 * @param {number} totalEpochs
 * @param {number} minLR
 */
export function warmupCosine(initialLR, warmupEpochs, totalEpochs, minLR = 0) {
  const cosine = cosineAnnealing(initialLR, totalEpochs - warmupEpochs, minLR);
  return warmup(initialLR, warmupEpochs, cosine);
}

/**
 * Cyclic learning rate: oscillate between bounds
 * @param {number} baseLR - Minimum LR
 * @param {number} maxLR - Maximum LR
 * @param {number} cycleLength - Epochs per cycle
 */
export function cyclicLR(baseLR, maxLR, cycleLength = 20) {
  return (epoch) => {
    const cycle = Math.floor(epoch / cycleLength);
    const x = Math.abs(2 * (epoch / cycleLength - cycle) - 1);
    return baseLR + (maxLR - baseLR) * Math.max(0, 1 - x);
  };
}

/**
 * Reduce on plateau: not epoch-based, but loss-based
 * Requires manual tracking of loss history.
 * @param {number} initialLR
 * @param {number} factor
 * @param {number} patience - Epochs to wait before reducing
 */
export function reduceLROnPlateau(initialLR, factor = 0.5, patience = 5) {
  let currentLR = initialLR;
  let bestLoss = Infinity;
  let waitCount = 0;

  return {
    getLR() { return currentLR; },
    step(loss) {
      if (loss < bestLoss * 0.999) {
        bestLoss = loss;
        waitCount = 0;
      } else {
        waitCount++;
        if (waitCount >= patience) {
          currentLR *= factor;
          waitCount = 0;
        }
      }
      return currentLR;
    }
  };
}
