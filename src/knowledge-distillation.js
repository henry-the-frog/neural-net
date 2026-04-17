// Knowledge distillation re-export with aliases
export { 
  softmaxWithTemp as softmaxWithTemperature,
  klDivergence,
  crossEntropy,
  distillationLoss,
  DistillationTrainer as KnowledgeDistillation,
  SimpleNetwork
} from './distillation.js';
