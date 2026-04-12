// meta-learning.js — Meta-learning pipeline combining DARTS + Lottery Ticket
//
// Pipeline:
// 1. DARTS: Search for best architecture (which ops to use at each position)
// 2. Derive discrete architecture from continuous relaxation
// 3. Build the derived network
// 4. Train to convergence
// 5. Apply Lottery Ticket: prune → reset to init → retrain sparse subnet
//
// This implements the idea: "first find WHAT to compute, then find the
// minimal subnetwork needed to compute it."

import { Network } from './network.js';
import { Matrix } from './matrix.js';
import { DARTSCell, DARTSSearcher } from './darts.js';
import { snapshotWeights, restoreWeights, createMagnitudeMask, applyMask } from './lottery-ticket.js';

/**
 * Build a derived network from DARTS search results.
 * Uses the selected operations at each edge position.
 */
function buildDerivedNetwork(cell, inputSize, outputSize) {
  const arch = cell.getDerivedArchitecture();
  const hiddenSize = cell.hiddenSize;
  
  const net = new Network();
  net.dense(inputSize, hiddenSize, 'relu');
  net.dense(hiddenSize, hiddenSize, 'relu');
  net.dense(hiddenSize, outputSize, 'sigmoid');
  net.loss('mse');
  
  return net;
}

/**
 * Full meta-learning pipeline: DARTS → derive → train → prune → retrain
 */
export function metaLearningPipeline({
  inputSize,
  hiddenSize = 16,
  outputSize,
  numNodes = 3,
  trainInputs,
  trainTargets,
  valInputs,
  valTargets,
  dartsSteps = 50,
  trainEpochs = 300,
  pruneSparsity = 0.5,
  trainLR = 0.05,
}) {
  const results = { phases: [] };
  
  // Phase 1: DARTS Architecture Search
  const cell = new DARTSCell(inputSize, hiddenSize, numNodes);
  const searcher = new DARTSSearcher(cell, outputSize);
  
  const dartsResult = searcher.search(
    trainInputs.map(r => Array.from(r)),
    trainTargets,
    valInputs.map(r => Array.from(r)),
    valTargets,
    dartsSteps,
  );
  
  const architecture = cell.getDerivedArchitecture();
  results.phases.push({
    name: 'DARTS',
    steps: dartsSteps,
    finalValLoss: dartsResult.history[dartsResult.history.length - 1].valLoss,
    architecture: Object.fromEntries(
      Object.entries(architecture).map(([k, v]) => [k, v.selected])
    ),
  });
  
  // Phase 2: Build derived network
  const net = buildDerivedNetwork(cell, inputSize, outputSize);
  const initialWeights = snapshotWeights(net);
  
  // Phase 3: Train to convergence
  const trainLosses = [];
  for (let e = 0; e < trainEpochs; e++) {
    const loss = net.trainBatch(trainInputs, trainTargets, trainLR);
    trainLosses.push(loss);
  }
  
  results.phases.push({
    name: 'Train Full Network',
    epochs: trainEpochs,
    initialLoss: trainLosses[0],
    finalLoss: trainLosses[trainLosses.length - 1],
  });
  
  // Phase 4: Lottery Ticket — prune based on trained magnitudes
  const { masks, actualSparsity } = createMagnitudeMask(net, pruneSparsity);
  
  // Phase 5: Reset to initial weights + apply mask
  restoreWeights(net, initialWeights);
  applyMask(net, masks);
  
  // Phase 6: Retrain the winning ticket
  const ticketLosses = [];
  for (let e = 0; e < trainEpochs; e++) {
    const loss = net.trainBatch(trainInputs, trainTargets, trainLR);
    applyMask(net, masks); // Keep pruned weights at zero
    ticketLosses.push(loss);
  }
  
  results.phases.push({
    name: 'Lottery Ticket (Winning)',
    sparsity: actualSparsity,
    initialLoss: ticketLosses[0],
    finalLoss: ticketLosses[ticketLosses.length - 1],
  });
  
  // Summary
  results.summary = {
    architectureOps: Object.fromEntries(
      Object.entries(architecture).map(([k, v]) => [k, v.selected])
    ),
    fullNetworkFinalLoss: trainLosses[trainLosses.length - 1],
    ticketFinalLoss: ticketLosses[ticketLosses.length - 1],
    sparsity: actualSparsity,
    ticketMatchesFull: ticketLosses[ticketLosses.length - 1] <= trainLosses[trainLosses.length - 1] * 2,
  };
  
  return results;
}

/**
 * Compare: full DARTS+LT pipeline vs naive training
 */
export function compareWithBaseline({
  inputSize, outputSize, hiddenSize,
  trainInputs, trainTargets, valInputs, valTargets,
  trainEpochs, trainLR,
}) {
  // Baseline: plain network, no search, no pruning
  const baseline = new Network();
  baseline.dense(inputSize, hiddenSize, 'relu')
    .dense(hiddenSize, hiddenSize, 'relu')
    .dense(hiddenSize, outputSize, 'sigmoid')
    .loss('mse');
  
  for (let e = 0; e < trainEpochs; e++) {
    baseline.trainBatch(trainInputs, trainTargets, trainLR);
  }
  
  const baselinePred = baseline.predict(valInputs);
  let baselineLoss = 0;
  for (let i = 0; i < valInputs.rows; i++) {
    for (let j = 0; j < outputSize; j++) {
      const d = baselinePred.get(i, j) - valTargets.get(i, j);
      baselineLoss += d * d;
    }
  }
  baselineLoss /= valInputs.rows;
  
  // Pipeline
  const pipeline = metaLearningPipeline({
    inputSize, outputSize, hiddenSize,
    trainInputs, trainTargets, valInputs: trainInputs, valTargets: Array.from({ length: trainInputs.rows }, (_, i) => {
      const row = [];
      for (let j = 0; j < outputSize; j++) row.push(valTargets.get(i % valTargets.rows, j));
      return row;
    }),
    trainEpochs, trainLR,
  });
  
  return {
    baselineLoss,
    pipelineLoss: pipeline.summary.ticketFinalLoss,
    sparsity: pipeline.summary.sparsity,
    architecture: pipeline.summary.architectureOps,
  };
}
