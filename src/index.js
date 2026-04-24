// index.js — Re-export all neural-net modules
//
// Usage:
//   import { Network, Matrix, ModelZoo, Datasets } from './src/index.js';

// Core
export { Network, Network as NeuralNetwork } from './network.js';
export { Matrix } from './matrix.js';
export * as autograd from './autograd.js';

// Compatibility aliases
import { sigmoid, tanh, relu, leakyRelu, softmax, linear, getActivation } from './activation.js';
import { mse, crossEntropy, getLoss } from './loss.js';
import { Network } from './network.js';
import { Dense } from './layer.js';

// Compatibility: wrap activations to accept scalar or array arguments
function wrapActivation(act) {
  const wrapped = { ...act };
  const origForward = wrapped.forward;
  if (origForward) {
    wrapped.forward = (x) => {
      if (typeof x === 'number') {
        const result = origForward([x]);
        return Array.isArray(result) ? result[0] : result;
      }
      return origForward(x);
    };
  }
  const origBackward = wrapped.backward;
  if (origBackward) {
    wrapped.backward = (x) => {
      if (typeof x === 'number') {
        const result = origBackward([x]);
        return Array.isArray(result) ? result[0] : result;
      }
      return origBackward(x);
    };
  }
  return wrapped;
}

export const activations = {
  sigmoid: wrapActivation(sigmoid),
  tanh: wrapActivation(tanh),
  relu: wrapActivation(relu),
  leakyRelu: wrapActivation(leakyRelu),
  softmax,
  linear: wrapActivation(linear)
};
export const losses = { mse, crossEntropy };
export function createNetwork(config, defaultActivation) {
  // Support createNetwork([2, 4, 1], 'sigmoid') — array of sizes + optional activation
  if (Array.isArray(config) && config.length > 0 && typeof config[0] === 'number') {
    const layers = [];
    const act = defaultActivation || 'sigmoid';
    for (let i = 1; i < config.length; i++) {
      layers.push(new Dense(config[i-1], config[i], act));
    }
    const net = new Network(layers);
    net.loss('mse');
    return net;
  }
  if (config.layers && config.layers.length > 0 && typeof config.layers[0] === 'object' && config.layers[0].size) {
    // Config-style layers: convert to actual layer objects
    const layers = [];
    for (let i = 1; i < config.layers.length; i++) {
      const layer = config.layers[i];
      const inputSize = config.layers[i-1].size;
      layers.push(new Dense(inputSize, layer.size, layer.activation || 'sigmoid'));
    }
    return new Network(layers);
  }
  return new Network(config);
}

// Layers
export { Dense, Dense as DenseLayer } from './layer.js';
export { Conv2D, MaxPool2D, Flatten } from './conv.js';
export { RNN, LSTM } from './rnn.js';
export { GRU } from './rnn.js';
export { EchoStateNetwork } from './esn.js';
export { MixtureOfExperts } from './moe.js';
export { MultiHeadAttention, SelfAttention } from './attention.js';
export { Embedding } from './embedding.js';

// Activations
export { sigmoid, tanh, relu, leakyRelu, softmax, linear, getActivation } from './activation.js';

// Models
export { ModelZoo } from './model-zoo.js';
export { Autoencoder } from './autoencoder.js';
export { VAE } from './vae.js';
export { GAN } from './gan.js';
export { MicroGPT, createSequences, decodeTokens, encodeText } from './microgpt.js';
export { PositionalEncoding, TransformerEncoderBlock } from './transformer.js';

// Data & Preprocessing
export { Datasets } from './datasets.js';
export { DataLoader, trainValTestSplit, stratifiedSplit } from './data-loader.js';
export { kFoldSplit as dataLoaderKFold } from './data-loader.js';
export { shuffle, normalize, applyNormalization, minMaxScale, createBatches } from './data.js';
export { StandardScaler, MinMaxScaler, oneHotEncode, trainTestSplit } from './preprocessing.js';
export { addNoise, randomFlipH, randomCrop, mixup, cutout, compose, randomBrightnessContrast } from './augmentation.js';

// Training
export { TrainingLogger, trainWithLogging } from './training-logger.js';
export { EarlyStopping, trainWithEarlyStopping } from './early-stopping.js';
export { ModelCheckpoint, TrainingState, ReduceLROnPlateau } from './model-checkpoint.js';
export { crossValidate, kFoldSplit } from './cross-validation.js';
export { autoML } from './automl.js';

// Loss functions
export { mse, crossEntropy, getLoss } from './loss.js';

// Evaluation
export {
  accuracy, confusionMatrix, precisionRecallF1, topKAccuracy,
  classificationReport, macroAverage, weightedAverage, microAverage,
  rocAuc, mae, rmse, r2Score, matthewsCorrelation, cohensKappa,
} from './metrics.js';
export { mse as metricsMse } from './metrics.js';

// Architecture Search
export { DARTSCell, DARTSSearcher, MixedOp } from './darts.js';
export { lotteryTicketExperiment, iterativePruning } from './lottery-ticket.js';
export { metaLearningPipeline } from './meta-learning.js';

// Advanced Layers
export { Residual, Sequential } from './residual.js';

// Normalizing Flows
export { NormalizingFlow, AffineCouplingLayer, PlanarFlow, ActNorm } from './normalizing-flows.js';

// Graph Neural Networks
export { Graph, GCNLayer, GNN, createKarateClub } from './gnn.js';

// Neuroscience-Inspired
export { HopfieldNetwork } from './hopfield.js';
export { SOM } from './som.js';
export { EnergyNetwork } from './ebm.js';
export { KANLayer, KAN } from './kan.js';

// Knowledge Distillation
export { crossEntropy as distillCrossEntropy } from './distillation.js';
