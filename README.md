# 🧠 Neural Net

A neural network library built from scratch in JavaScript. Zero dependencies. 390+ tests.

## Features

### Layers
- **Dense** (fully connected) — configurable activations
- **Conv2D** — 2D convolution with stride, padding
- **MaxPool2D** — max pooling
- **BatchNorm** — batch normalization
- **Dropout** — regularization
- **Flatten** — reshape for Dense after Conv
- **RNN** — vanilla recurrent neural network
- **LSTM** — Long Short-Term Memory
- **GRU** — Gated Recurrent Unit
- **Transformer** — multi-head attention encoder
- **Embedding** — word/token embeddings
- **LayerNorm** — layer normalization
- **Residual** — skip connections

### Activations
sigmoid, tanh, relu, leaky_relu, elu, swish, gelu, softmax, linear, softplus

### Optimizers
SGD (with momentum), Adam, RMSprop, AdaGrad

### Loss Functions
MSE, Cross-Entropy, Binary Cross-Entropy, Huber, Hinge

### Training
- Mini-batch gradient descent
- Learning rate schedulers: step decay, exponential, cosine annealing, warmup, cyclic, reduce-on-plateau
- Early stopping
- Gradient clipping
- L1/L2 regularization

### Data Utilities
- `shuffle()` — Fisher-Yates with aligned targets
- `trainTestSplit()` — random train/test split
- `normalize()` / `minMaxScale()` — feature scaling
- `addNoise()` — Gaussian augmentation
- `createBatches()` — mini-batch creation
- `oneHotEncode()` — label encoding

### Evaluation Metrics
- Classification: accuracy, precision, recall, F1, confusion matrix
- Regression: MSE, MAE, RMSE, R²

### Model Persistence
- `network.save()` — serialize to JSON
- `Network.load(json)` — restore from JSON

## Quick Start

```javascript
import { Network, Matrix } from './src/index.js';

// XOR classifier
const net = new Network();
net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');

const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
const targets = Matrix.fromArray([[0],[1],[1],[0]]);

net.train({ inputs, targets }, { epochs: 1000, learningRate: 0.5 });

console.log(net.predict(inputs).toArray());
// [[0.01], [0.99], [0.99], [0.01]]
```

## CNN Digit Classifier

```javascript
import { Network, Conv2D, MaxPool2D, Flatten, generateDigitDataset } from './src/index.js';

const { inputs, targets } = generateDigitDataset(50);

const net = new Network();
net.add(new Conv2D(5, 5, 1, 4, 3, 'relu'));
net.add(new Flatten());
net.dense(36, 10, 'softmax');
net.loss('cross_entropy');

net.train({ inputs, targets }, { epochs: 100, learningRate: 0.01 });
```

## LSTM Sequence Prediction

```javascript
import { Network, LSTM, Matrix } from './src/index.js';

const net = new Network();
net.add(new LSTM(1, 8));
net.dense(8, 1, 'linear');
net.loss('mse');

// Predict sum of sequence
const seqs = Matrix.fromArray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
const targets = Matrix.fromArray([[0.2], [0.5]]);

net.train({ inputs: seqs, targets }, { epochs: 500, learningRate: 0.01 });
```

## Save & Load

```javascript
// Save trained model
const json = net.save();
fs.writeFileSync('model.json', json);

// Load later
const loaded = Network.load(fs.readFileSync('model.json', 'utf8'));
loaded.predict(input);
```

## Architecture

```
src/
├── network.js       # Network class (train, predict, save/load)
├── layer.js         # Dense layer
├── conv.js          # Conv2D, MaxPool2D, Flatten
├── rnn.js           # RNN, LSTM, GRU
├── transformer.js   # TransformerEncoderBlock
├── batchnorm.js     # BatchNorm
├── activation.js    # All activation functions
├── loss.js          # Loss functions
├── optimizer.js     # SGD, Adam, RMSprop, AdaGrad
├── matrix.js        # Matrix math (Float64Array-backed)
├── scheduler.js     # Learning rate schedulers
├── data.js          # Data utilities (shuffle, split, normalize)
├── metrics.js       # Evaluation metrics
├── augmentation.js  # Data augmentation
├── digits.js        # Synthetic digit dataset
├── regularization.js # L1/L2, early stopping
└── index.js         # Public API
```

## Stats

- **37 source files**, ~9,000 lines
- **390+ tests**, all passing
- **Zero dependencies** — pure JavaScript
- Verified with numerical gradient checking (finite differences)

## Gradient Verification

Every layer's backward pass has been verified against numerical gradients:
- Dense: sigmoid, relu, tanh — weights, biases, input gradients
- Conv2D: filters, biases, input, stride+padding, multi-channel
- RNN: Wih, Whh, bias (BPTT through time)
- LSTM: all 4 gate weights + biases
- GRU: Wz, Wr, Wh, bias
- BatchNorm: gamma, beta

## License

MIT
