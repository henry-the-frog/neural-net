// gradient-checkpoint.js — Gradient Checkpointing for memory-efficient training
// Paper: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
//
// Key idea: Instead of storing all intermediate activations for backprop,
// only store "checkpoints" at selected layers. During backward pass,
// recompute activations from the nearest checkpoint.
//
// Memory reduction: O(L) → O(√L) activations stored, at cost of ~33% more compute.

import { Matrix } from './matrix.js';

/**
 * A checkpointed segment of layers.
 * During forward: only stores input/output (not intermediate activations).
 * During backward: recomputes forward from checkpoint to get activations.
 */
export class CheckpointSegment {
  constructor(layers) {
    this.layers = layers;
    this.checkpointInput = null; // Saved for recomputation
    this.checkpointOutput = null;
  }

  forward(input) {
    // Save input for potential recomputation during backward
    this.checkpointInput = input;
    
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    
    this.checkpointOutput = x;
    return x;
  }

  backward(dOutput) {
    if (!this.checkpointInput) {
      throw new Error('Must call forward() before backward()');
    }
    
    // Recompute forward to restore intermediate activations
    let x = this.checkpointInput;
    const activations = [x];
    for (const layer of this.layers) {
      x = layer.forward(x);
      activations.push(x);
    }
    
    // Now do backward through all layers
    let dX = dOutput;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      dX = this.layers[i].backward(dX);
    }
    
    return dX;
  }

  update(lr) {
    for (const layer of this.layers) {
      if (layer.update) layer.update(lr);
    }
  }

  paramCount() {
    return this.layers.reduce((sum, l) => sum + (l.paramCount?.() || 0), 0);
  }
}

/**
 * Wrap a sequence of layers into checkpointed segments.
 * Optimal segment size for N layers is √N.
 * 
 * @param {Array} layers - All layers to checkpoint
 * @param {number} segmentSize - Number of layers per segment (default: √N)
 * @returns {Array<CheckpointSegment>} Checkpointed segments
 */
export function checkpoint(layers, segmentSize = null) {
  const N = layers.length;
  const size = segmentSize || Math.max(1, Math.round(Math.sqrt(N)));
  
  const segments = [];
  for (let i = 0; i < N; i += size) {
    const end = Math.min(i + size, N);
    segments.push(new CheckpointSegment(layers.slice(i, end)));
  }
  
  return segments;
}

/**
 * Compute memory savings from checkpointing.
 * @param {number} numLayers - Total number of layers
 * @param {number} segmentSize - Layers per segment
 * @returns {object} { storedActivations, withoutCheckpointing, savings }
 */
export function memoryEstimate(numLayers, segmentSize = null) {
  const size = segmentSize || Math.round(Math.sqrt(numLayers));
  const numSegments = Math.ceil(numLayers / size);
  
  // Without checkpointing: store all N activations
  const without = numLayers;
  
  // With checkpointing: store segment boundaries (numSegments + 1)
  // plus recomputed activations within one segment during backward (size)
  const with_ = numSegments + 1 + size;
  
  return {
    storedActivations: with_,
    withoutCheckpointing: without,
    savings: `${((1 - with_ / without) * 100).toFixed(1)}%`,
    segmentSize: size,
    numSegments,
  };
}
