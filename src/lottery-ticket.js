// lottery-ticket.js — Lottery Ticket Hypothesis
//
// Frankle & Carlin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
// Neural Networks" (ICLR 2019)
//
// Core idea: Dense networks contain sparse subnetworks ("winning tickets") that,
// when trained in isolation from their original initialization, can match the
// full network's accuracy.
//
// Algorithm:
// 1. Randomly initialize network f(x; θ₀)
// 2. Train to get θ_trained
// 3. Prune p% of weights (smallest magnitude) → create mask m
// 4. Reset remaining weights to θ₀ (original init)
// 5. Retrain f(x; m ⊙ θ₀) → this is the "winning ticket"
//
// If the winning ticket converges to similar accuracy, the hypothesis holds:
// the lottery ticket (subnetwork + initialization) was present from the start.

import { Network } from './network.js';
import { Matrix } from './matrix.js';

/**
 * Save a snapshot of all layer weights (deep copy)
 */
export function snapshotWeights(net) {
  return net.layers.map(layer => {
    const snap = {};
    if (layer.weights) {
      snap.weights = new Float64Array(layer.weights.data);
      snap.wRows = layer.weights.rows;
      snap.wCols = layer.weights.cols;
    }
    if (layer.biases) {
      snap.biases = new Float64Array(layer.biases.data);
      snap.bRows = layer.biases.rows;
      snap.bCols = layer.biases.cols;
    }
    return snap;
  });
}

/**
 * Restore layer weights from a snapshot
 */
export function restoreWeights(net, snapshot) {
  for (let i = 0; i < net.layers.length; i++) {
    const snap = snapshot[i];
    if (snap.weights && net.layers[i].weights) {
      net.layers[i].weights = new Matrix(snap.wRows, snap.wCols, new Float64Array(snap.weights));
    }
    if (snap.biases && net.layers[i].biases) {
      net.layers[i].biases = new Matrix(snap.bRows, snap.bCols, new Float64Array(snap.biases));
    }
  }
}

/**
 * Create a pruning mask based on weight magnitudes.
 * Returns masks for each layer (1 = keep, 0 = pruned).
 */
export function createMagnitudeMask(net, sparsity) {
  // Collect all weight magnitudes
  const allMags = [];
  for (const layer of net.layers) {
    if (layer.weights) {
      for (let i = 0; i < layer.weights.data.length; i++) {
        allMags.push(Math.abs(layer.weights.data[i]));
      }
    }
  }
  
  // Find global threshold
  allMags.sort((a, b) => a - b);
  const threshold = allMags[Math.floor(allMags.length * sparsity)];
  
  // Create masks
  const masks = net.layers.map(layer => {
    if (!layer.weights) return null;
    const mask = new Float64Array(layer.weights.data.length);
    for (let i = 0; i < layer.weights.data.length; i++) {
      mask[i] = Math.abs(layer.weights.data[i]) > threshold ? 1 : 0;
    }
    return mask;
  });
  
  // Count actual sparsity
  let total = 0, pruned = 0;
  for (const m of masks) {
    if (!m) continue;
    total += m.length;
    for (let i = 0; i < m.length; i++) {
      if (m[i] === 0) pruned++;
    }
  }
  
  return { masks, threshold, actualSparsity: total > 0 ? pruned / total : 0 };
}

/**
 * Apply a mask to network weights (zero out pruned weights)
 */
export function applyMask(net, masks) {
  for (let i = 0; i < net.layers.length; i++) {
    if (!masks[i] || !net.layers[i].weights) continue;
    for (let j = 0; j < net.layers[i].weights.data.length; j++) {
      net.layers[i].weights.data[j] *= masks[i][j];
    }
  }
}

/**
 * Run a Lottery Ticket experiment.
 * 
 * @param {Function} createNetwork - Factory: () => configured Network
 * @param {Matrix} trainInputs - Training data
 * @param {Matrix} trainTargets - Training targets
 * @param {Object} opts - Options
 * @returns {Object} Results with full/ticket metrics
 */
export function lotteryTicketExperiment({
  createNetwork,
  trainInputs,
  trainTargets,
  trainEpochs = 500,
  trainLR = 0.1,
  sparsity = 0.5,
  retrainEpochs = null,
  retrainLR = null,
}) {
  retrainEpochs = retrainEpochs || trainEpochs;
  retrainLR = retrainLR || trainLR;
  
  // Step 1: Create and save initial weights
  const net = createNetwork();
  const initialWeights = snapshotWeights(net);
  
  // Step 2: Train to completion
  const fullLosses = [];
  for (let e = 0; e < trainEpochs; e++) {
    const loss = net.trainBatch(trainInputs, trainTargets, trainLR);
    fullLosses.push(loss);
  }
  const fullFinalLoss = fullLosses[fullLosses.length - 1];
  
  // Step 3: Prune — create mask based on trained weight magnitudes
  const { masks, actualSparsity } = createMagnitudeMask(net, sparsity);
  
  // Step 4: Reset to initial weights and apply mask
  restoreWeights(net, initialWeights);
  applyMask(net, masks);
  
  // Step 5: Retrain the winning ticket
  const ticketLosses = [];
  for (let e = 0; e < retrainEpochs; e++) {
    const loss = net.trainBatch(trainInputs, trainTargets, retrainLR);
    // Re-apply mask after each training step to keep pruned weights at zero
    applyMask(net, masks);
    ticketLosses.push(loss);
  }
  const ticketFinalLoss = ticketLosses[ticketLosses.length - 1];
  
  // Step 6: Random ticket (control) — random init + same mask
  const randomNet = createNetwork();
  applyMask(randomNet, masks);
  const randomLosses = [];
  for (let e = 0; e < retrainEpochs; e++) {
    const loss = randomNet.trainBatch(trainInputs, trainTargets, retrainLR);
    applyMask(randomNet, masks);
    randomLosses.push(loss);
  }
  const randomFinalLoss = randomLosses[randomLosses.length - 1];
  
  return {
    sparsity: actualSparsity,
    fullNetwork: {
      finalLoss: fullFinalLoss,
      losses: fullLosses,
    },
    winningTicket: {
      finalLoss: ticketFinalLoss,
      losses: ticketLosses,
    },
    randomTicket: {
      finalLoss: randomFinalLoss,
      losses: randomLosses,
    },
    // The hypothesis holds if the winning ticket performs close to the full network
    // and better than the random ticket
    hypothesisHolds: ticketFinalLoss <= fullFinalLoss * 2 && ticketFinalLoss < randomFinalLoss,
  };
}

/**
 * Iterative Magnitude Pruning (IMP)
 * Gradually increases sparsity over multiple rounds.
 */
export function iterativePruning({
  createNetwork,
  trainInputs,
  trainTargets,
  trainEpochs = 300,
  trainLR = 0.1,
  rounds = 5,
  prunePerRound = 0.2,
}) {
  const results = [];
  let cumulativeSparsity = 0;
  let currentMasks = null;
  let currentInitialWeights = null;
  
  for (let round = 0; round < rounds; round++) {
    const net = createNetwork();
    
    // First round: save initial weights
    if (round === 0) {
      currentInitialWeights = snapshotWeights(net);
    } else {
      // Restore original initialization
      restoreWeights(net, currentInitialWeights);
    }
    
    // Apply existing mask
    if (currentMasks) {
      applyMask(net, currentMasks);
    }
    
    // Train
    for (let e = 0; e < trainEpochs; e++) {
      net.trainBatch(trainInputs, trainTargets, trainLR);
      if (currentMasks) applyMask(net, currentMasks);
    }
    
    const finalLoss = net.trainBatch(trainInputs, trainTargets, trainLR);
    
    // Prune additional weights
    cumulativeSparsity = 1 - Math.pow(1 - prunePerRound, round + 1);
    const { masks, actualSparsity } = createMagnitudeMask(net, cumulativeSparsity);
    
    // Merge with existing masks (intersection: a weight stays only if kept in both)
    if (currentMasks) {
      for (let i = 0; i < masks.length; i++) {
        if (!masks[i] || !currentMasks[i]) continue;
        for (let j = 0; j < masks[i].length; j++) {
          masks[i][j] *= currentMasks[i][j];
        }
      }
    }
    currentMasks = masks;
    
    results.push({
      round: round + 1,
      targetSparsity: cumulativeSparsity,
      actualSparsity,
      finalLoss,
    });
  }
  
  return results;
}
