# neural-net

A comprehensive neural network library built from scratch in JavaScript. No dependencies. 71 source modules, 100 test files, 1230+ tests, ~15,600 lines of code.

## Quick Start

```javascript
import { Network } from './src/network.js';
import { Matrix } from './src/matrix.js';

const net = new Network();
net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');

const inputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targets = Matrix.fromArray([[0], [1], [1], [0]]);

for (let i = 0; i < 1000; i++) {
  net.trainBatch(inputs, targets, 0.5);
}

console.log(net.predict(inputs)); // XOR!
```

## Modules

### Core
| Module | Description |
|--------|------------|
| `network.js` | Network builder, training, serialization |
| `matrix.js` | Matrix operations (multiply, transpose, map) |
| `loss.js` | MSE, cross-entropy loss functions |
| `autograd.js` | Automatic differentiation (15 ops, backward pass) |

### Layers
| Module | Description |
|--------|------------|
| `dense.js` | Fully connected layer |
| `conv2d.js` | 2D convolution (CNN) |
| `pool.js` | Max/Average pooling |
| `flatten.js` | Flatten for CNN→Dense transition |
| `dropout.js` | Training regularization |
| `batchnorm.js` | Batch normalization |

### Activations
| Module | Description |
|--------|------------|
| `activation.js` | ReLU, sigmoid, tanh, softmax, GELU, swish, ELU, SELU, leaky ReLU |

### Optimizers
| Module | Description |
|--------|------------|
| `optimizer.js` | SGD, Adam, RMSprop, AdaGrad |
| `lr-scheduler.js` | Step, cosine, warmup, one-cycle LR schedules |

### Advanced Architectures
| Module | Description |
|--------|------------|
| `transformer.js` | Encoder/decoder blocks, positional encoding, layer norm |
| `attention.js` | Multi-head attention, scaled dot-product attention |
| `lstm.js` | Long Short-Term Memory |
| `gru.js` | Gated Recurrent Unit |
| `esn.js` | Echo State Network (reservoir computing) |
| `capsule.js` | Capsule Network with dynamic routing |
| `gnn.js` | Graph Neural Network (GCN, Karate Club) |
| `hypernetwork.js` | Task-conditioned hypernetwork |
| `moe.js` | Mixture of Experts with top-k gating |

### Neuroscience-Inspired
| Module | Description |
|--------|------------|
| `snn.js` | Spiking Neural Network (LIF, Izhikevich, STDP) |
| `hopfield.js` | Hopfield Network (energy-based associative memory) |
| `som.js` | Self-Organizing Map |
| `ebm.js` | Energy-Based Model with Langevin sampling |

### Modern ML Research
| Module | Description |
|--------|------------|
| `kan.js` | Kolmogorov-Arnold Network (B-spline activations) |
| `normalizing-flows.js` | Planar flow, affine coupling, ActNorm |
| `darts.js` | DARTS — Differentiable Architecture Search |
| `lottery-ticket.js` | Lottery Ticket Hypothesis (magnitude pruning) |
| `meta-learning.js` | DARTS + Lottery Ticket meta-learning pipeline |
| `ntm.js` | Neural Turing Machine (external memory) |
| `neuroevolution.js` | Genetic algorithm, tournament/roulette selection |

### Utilities
| Module | Description |
|--------|------------|
| `model-zoo.js` | Pre-configured architectures (XOR, classifier, autoencoder, etc.) |
| `training-logger.js` | Metrics tracking, ASCII charts, JSON/CSV export |
| `datasets.js` | Synthetic datasets (spiral, moons, circles, blobs, sine) |
| `preprocessing.js` | StandardScaler, MinMaxScaler, oneHotEncode, trainTestSplit |
| `cross-validation.js` | K-fold cross-validation with accuracy and loss |
| `early-stopping.js` | Patience-based training termination, best model restore |
| `metrics.js` | Confusion matrix, precision, recall, F1, classification report |
| `pruning.js` | Network pruning (structured, unstructured) |
| `quantization.js` | Weight quantization (int8, float16) |
| `data-augmentation.js` | Image transforms for training |

### Applications
| Module | Description |
|--------|------------|
| `mnist.js` | MNIST digit recognition demo |
| `reinforcement.js` | REINFORCE policy gradient |

## Testing

```bash
# Run all tests
npm test

# Run specific test file
node --test test/autograd.test.js

# Run stress tests
node --test test/autograd-stress.test.js
```

Every module has both unit tests and deep stress tests that verify mathematical properties:
- Numerical gradient agreement (analytical vs finite-difference)
- Invariants (energy monotonicity, weight symmetry, probability normalization)
- Boundary conditions (zero input, extreme values)
- Convergence (loss decreasing, function approximation)

## Architecture

```
src/
├── Core:       network.js, matrix.js, loss.js, autograd.js
├── Layers:     dense.js, conv2d.js, pool.js, flatten.js
├── Training:   optimizer.js, lr-scheduler.js, dropout.js, batchnorm.js
├── Sequences:  lstm.js, gru.js, esn.js, transformer.js, attention.js
├── Graphs:     gnn.js, capsule.js, moe.js, hypernetwork.js
├── Neuro:      snn.js, hopfield.js, som.js, ebm.js
├── Research:   kan.js, normalizing-flows.js, darts.js, lottery-ticket.js
└── Utils:      model-zoo.js, training-logger.js, pruning.js
```

## Philosophy

- **Zero dependencies**: Everything built from scratch
- **Depth over breadth**: Every module has stress tests verifying mathematical correctness
- **Educational**: Clean code that's readable and follows the papers
- **Practical**: ModelZoo + TrainingLogger for real usage

## License

MIT
