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

export const activations = { sigmoid, tanh, relu, leakyRelu, softmax, linear };
export const losses = { mse, crossEntropy };
export function createNetwork(config) { return new Network(config); }

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
export { MicroGPT, createSequences, decodeTokens } from './microgpt.js';
export { PositionalEncoding } from './transformer.js';

// Data & Preprocessing
export { Datasets } from './datasets.js';
export { StandardScaler, MinMaxScaler, oneHotEncode, trainTestSplit } from './preprocessing.js';
export { addNoise, randomFlipH, randomCrop, mixup, cutout, compose, randomBrightnessContrast } from './augmentation.js';

// Training
export { TrainingLogger, trainWithLogging } from './training-logger.js';
export { EarlyStopping, trainWithEarlyStopping } from './early-stopping.js';
export { crossValidate, kFoldSplit } from './cross-validation.js';
export { autoML } from './automl.js';

// Loss functions
export { mse, crossEntropy, getLoss } from './loss.js';

// Evaluation
export {
  confusionMatrix, precision, recall, f1Score, accuracy,
  classificationReport, printConfusionMatrix,
} from './metrics.js';

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
