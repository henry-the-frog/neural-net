// index.js — Re-export all neural-net modules
//
// Usage:
//   import { Network, Matrix, ModelZoo, Datasets } from './src/index.js';

// Core
export { Network } from './network.js';
export { Matrix } from './matrix.js';
export * as autograd from './autograd.js';

// Data & Preprocessing
export { Datasets } from './datasets.js';
export { StandardScaler, MinMaxScaler, oneHotEncode, trainTestSplit } from './preprocessing.js';

// Models
export { ModelZoo } from './model-zoo.js';

// Training
export { TrainingLogger, trainWithLogging } from './training-logger.js';
export { EarlyStopping, trainWithEarlyStopping } from './early-stopping.js';
export { crossValidate, kFoldSplit } from './cross-validation.js';
export { autoML } from './automl.js';

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
export { LSTM } from './lstm.js';
export { GRU } from './gru.js';
export { EchoStateNetwork } from './esn.js';
export { MixtureOfExperts } from './moe.js';

// Normalizing Flows
export { NormalizingFlow, AffineCouplingLayer, PlanarFlow, ActNorm } from './normalizing-flows.js';

// Graph Neural Networks
export { Graph, GCNLayer, GNN, createKarateClub } from './gnn.js';

// Neuroscience-Inspired
export { HopfieldNetwork } from './hopfield.js';
export { SOM } from './som.js';
export { EnergyNetwork } from './ebm.js';
export { KANLayer, KAN } from './kan.js';
