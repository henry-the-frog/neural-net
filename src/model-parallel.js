// model-parallel.js — Model parallelism concepts (simulated)
// Tensor parallelism: split weight matrices across "devices"
// Pipeline parallelism: split layers across "devices"

export function tensorParallelSplit(weights, numDevices) {
  const totalCols = weights.length;
  const colsPerDevice = Math.ceil(totalCols / numDevices);
  const shards = [];
  for (let d = 0; d < numDevices; d++) {
    const start = d * colsPerDevice;
    const end = Math.min(start + colsPerDevice, totalCols);
    shards.push(weights.slice(start, end));
  }
  return shards;
}

export function tensorParallelGather(shards) {
  return shards.flat();
}

export function pipelineStages(layers, numStages) {
  const layersPerStage = Math.ceil(layers.length / numStages);
  const stages = [];
  for (let s = 0; s < numStages; s++) {
    const start = s * layersPerStage;
    stages.push(layers.slice(start, start + layersPerStage));
  }
  return stages;
}

export function pipelineBubbleRatio(numMicroBatches, numStages) {
  // Bubble time / total time
  return (numStages - 1) / (numMicroBatches + numStages - 1);
}
