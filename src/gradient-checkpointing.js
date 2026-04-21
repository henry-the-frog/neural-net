// gradient-checkpointing.js — Memory-Efficient Training via Gradient Checkpointing
// Paper: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
//
// Standard training: stores all intermediate activations for backward pass → O(N) memory
// Gradient checkpointing: only stores checkpoints at certain layers, recomputes
// activations between checkpoints during backward → O(√N) memory at ~33% more compute
//
// This is how large models are trained on limited GPU memory.

/**
 * Simulate gradient checkpointing memory savings.
 *
 * @param {number} numLayers - total decoder layers
 * @param {number} numCheckpoints - number of checkpoint boundaries
 * @param {number} activationSizePerLayer - memory per layer's activations
 * @returns {object} memory comparison
 */
export function checkpointingAnalysis(numLayers, numCheckpoints, activationSizePerLayer) {
  // Standard: store all layer activations
  const standardMemory = numLayers * activationSizePerLayer;

  // Checkpointing: store only checkpoints + recompute between them
  // Memory = checkpoints * activationSize + segment * activationSize
  // where segment = numLayers / numCheckpoints (max activations between checkpoints)
  const segmentSize = Math.ceil(numLayers / numCheckpoints);
  const checkpointMemory = (numCheckpoints + segmentSize) * activationSizePerLayer;

  // Compute overhead: recompute each segment once during backward
  const recomputeFraction = 1 - 1 / segmentSize; // fraction of compute redone
  const computeOverhead = recomputeFraction * 100;

  return {
    standardMemory,
    checkpointMemory,
    memorySavings: ((1 - checkpointMemory / standardMemory) * 100).toFixed(1) + '%',
    computeOverhead: computeOverhead.toFixed(1) + '%',
    numCheckpoints,
    segmentSize,
    optimal: numCheckpoints === Math.floor(Math.sqrt(numLayers)),
  };
}

/**
 * Find optimal number of checkpoints (minimizes peak memory).
 * Optimal: √N checkpoints for O(√N) memory.
 */
export function optimalCheckpoints(numLayers) {
  return Math.max(1, Math.floor(Math.sqrt(numLayers)));
}

/**
 * Checkpoint schedule: which layers to checkpoint.
 *
 * @param {number} numLayers
 * @param {number} numCheckpoints
 * @returns {number[]} layer indices to checkpoint
 */
export function checkpointSchedule(numLayers, numCheckpoints) {
  const step = Math.ceil(numLayers / numCheckpoints);
  const schedule = [];
  for (let i = 0; i < numLayers; i += step) {
    schedule.push(i);
  }
  return schedule;
}

/**
 * Simulate checkpointed forward pass.
 * Returns which activations are stored vs recomputed.
 *
 * @param {number} numLayers
 * @param {number[]} checkpointLayers - which layers to checkpoint
 * @returns {{ stored: number[], recomputed: number[], peakMemory: number }}
 */
export function simulateCheckpointedPass(numLayers, checkpointLayers) {
  const cpSet = new Set(checkpointLayers);
  const stored = [];  // layers whose activations are kept
  const recomputed = []; // layers recomputed during backward

  // Forward: only store checkpoint activations
  for (let i = 0; i < numLayers; i++) {
    if (cpSet.has(i)) stored.push(i);
  }

  // Backward: between checkpoints, recompute
  const sortedCPs = [...checkpointLayers].sort((a, b) => a - b);
  for (let c = 0; c < sortedCPs.length; c++) {
    const start = sortedCPs[c];
    const end = c + 1 < sortedCPs.length ? sortedCPs[c + 1] : numLayers;
    for (let i = start + 1; i < end; i++) {
      recomputed.push(i);
    }
  }

  // Peak memory: checkpoints + one full segment
  const maxSegment = computeMaxSegment(numLayers, sortedCPs);
  const peakMemory = stored.length + maxSegment;

  return { stored, recomputed, peakMemory };
}

function computeMaxSegment(numLayers, sortedCPs) {
  if (sortedCPs.length === 0) return numLayers;
  let max = 0;
  for (let i = 0; i < sortedCPs.length; i++) {
    const end = i + 1 < sortedCPs.length ? sortedCPs[i + 1] : numLayers;
    max = Math.max(max, end - sortedCPs[i]);
  }
  return max;
}
