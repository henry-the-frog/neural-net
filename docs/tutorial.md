# Training Your First Neural Network

*A hands-on guide to building, training, and understanding neural networks — from scratch, in JavaScript.*

## Prerequisites

- Basic JavaScript knowledge
- `node` v20+ installed
- Clone this repo and run `npm install`

## 1. The Building Block: Matrices

Neural networks are fundamentally about matrix math. Every piece of data, every weight, every gradient is a matrix.

```javascript
import { Matrix } from './src/matrix.js';

// Create a 2x3 matrix
const m = new Matrix(2, 3);
m.set(0, 0, 1.0);  // row 0, col 0
m.set(0, 1, 2.0);
m.set(1, 0, 3.0);

console.log(m.toString());
// [1.00, 2.00, 0.00]
// [3.00, 0.00, 0.00]

// Create from arrays (more common)
const data = Matrix.fromArray([[1, 2, 3], [4, 5, 6]]);
console.log(data.rows, data.cols); // 2, 3
```

## 2. Building a Network

A neural network is a stack of layers. Each layer transforms its input through weights, biases, and an activation function.

```javascript
import { Network } from './src/network.js';

const net = new Network();

// Add layers: dense(inputSize, outputSize, activation)
net.dense(2, 8, 'relu');     // Input: 2 features → 8 neurons
net.dense(8, 4, 'relu');     // Hidden: 8 → 4 neurons
net.dense(4, 1, 'sigmoid');  // Output: 4 → 1 neuron (0-1 probability)

// Set the loss function
net.loss('mse');  // Mean Squared Error

// See what you built
console.log(net.summary());
```

**Activation functions** add non-linearity — without them, stacking layers would be pointless (a stack of linear transforms is just one linear transform):

| Function | Range | Best for |
|----------|-------|----------|
| `relu` | [0, ∞) | Hidden layers (default choice) |
| `sigmoid` | (0, 1) | Binary classification output |
| `tanh` | (-1, 1) | When you need negative outputs |
| `softmax` | (0, 1), sums to 1 | Multi-class classification |

## 3. The Forward Pass

When you feed data into a network, it flows forward through each layer:

```
Input → [Layer 1: weights × input + bias → activation] → [Layer 2: ...] → Output
```

```javascript
import { Network, Matrix } from './src/index.js';

const net = new Network();
net.dense(2, 4, 'relu');
net.dense(4, 1, 'sigmoid');
net.loss('mse');

// Feed an input through the network
const input = Matrix.fromArray([[0.5, 0.8]]);
const output = net.predict(input);
console.log('Prediction:', output.get(0, 0));
// Something random — the network hasn't learned anything yet!
```

## 4. Loss Functions: Measuring Error

The **loss function** tells the network how wrong its predictions are. Lower = better.

```javascript
// Mean Squared Error — good for regression
net.loss('mse');
// Formula: (1/n) × Σ(predicted - actual)²

// Cross-Entropy — good for classification
net.loss('cross-entropy');
// Formula: -Σ(actual × log(predicted))
```

## 5. Backpropagation: Learning from Mistakes

This is where the magic happens. Backpropagation computes how much each weight contributed to the error, then adjusts weights to reduce that error.

The process:
1. **Forward pass**: compute output
2. **Compute loss**: how wrong was the output?
3. **Backward pass**: compute gradients (∂loss/∂weight for every weight)
4. **Update weights**: weights -= learning_rate × gradients

You don't need to implement this yourself — the `Network` class handles it:

```javascript
// Single training step
const loss = net.trainBatch(inputs, targets, learningRate);
console.log('Loss:', loss);
```

## 6. The Training Loop

Training means repeating forward→loss→backward→update thousands of times:

```javascript
import { Network, Matrix } from './src/index.js';

// XOR — the classic neural network challenge
const inputs = Matrix.fromArray([[0,0], [0,1], [1,0], [1,1]]);
const targets = Matrix.fromArray([[0], [1], [1], [0]]);

const net = new Network();
net.dense(2, 8, 'tanh');
net.dense(8, 1, 'sigmoid');
net.loss('mse');

// Train for 2000 epochs
for (let epoch = 0; epoch < 2000; epoch++) {
  const loss = net.trainBatch(inputs, targets, 0.5);
  if (epoch % 500 === 0) {
    console.log(`Epoch ${epoch}: loss = ${loss.toFixed(4)}`);
  }
}

// Test
console.log('\nPredictions:');
for (let i = 0; i < 4; i++) {
  const row = inputs.slice(i, i + 1);
  const pred = net.predict(row).get(0, 0);
  const actual = targets.get(i, 0);
  console.log(`  [${inputs.get(i,0)}, ${inputs.get(i,1)}] → ${pred.toFixed(3)} (expected ${actual})`);
}
```

Expected output:
```
Epoch 0: loss = 0.2513
Epoch 500: loss = 0.0142
Epoch 1000: loss = 0.0028
Epoch 1500: loss = 0.0011

Predictions:
  [0, 0] → 0.023 (expected 0)
  [0, 1] → 0.971 (expected 1)
  [1, 0] → 0.968 (expected 1)
  [1, 1] → 0.034 (expected 0)
```

## 7. Using the Built-in Trainer

For more control, use the `train()` method with options:

```javascript
const history = net.train({ inputs, targets }, {
  epochs: 1000,
  learningRate: 0.1,
  momentum: 0.9,       // Helps escape local minima
  batchSize: 32,       // Mini-batch gradient descent
  verbose: true,       // Print progress
  lrSchedule: 'cosine' // Learning rate decay
});

console.log('Final loss:', history[history.length - 1]);
```

### Training Tips

- **Learning rate too high?** Loss oscillates or diverges. Try 0.01 or 0.001.
- **Learning rate too low?** Loss decreases very slowly. Try 0.1 or 0.5.
- **Network too small?** Can't learn the pattern. Add more neurons or layers.
- **Network too big?** Overfits (memorizes instead of generalizing). Add dropout or reduce size.
- **Loss stuck?** Try different activation functions, add momentum, or increase learning rate.

## 8. Example: Function Approximation

Neural networks are universal approximators — they can learn any continuous function. Let's teach one to approximate sin(x):

```javascript
import { Network, Matrix } from './src/index.js';

// Generate training data
const N = 200;
const inputs = new Matrix(N, 1);
const targets = new Matrix(N, 1);
for (let i = 0; i < N; i++) {
  const x = (i / N) * 2 * Math.PI;
  inputs.set(i, 0, x / (2 * Math.PI));     // Normalize input to [0, 1]
  targets.set(i, 0, (Math.sin(x) + 1) / 2); // Normalize output to [0, 1]
}

const net = new Network();
net.dense(1, 32, 'tanh');
net.dense(32, 16, 'tanh');
net.dense(16, 1, 'sigmoid');
net.loss('mse');

const history = net.train({ inputs, targets }, {
  epochs: 500,
  learningRate: 0.5,
  momentum: 0.9,
  batchSize: 32,
  lrSchedule: 'cosine'
});

console.log(`Final loss: ${history[history.length - 1].toFixed(6)}`);

// Test predictions
for (let i = 0; i <= 4; i++) {
  const x = (i / 4) * 2 * Math.PI;
  const actual = Math.sin(x);
  const predicted = net.predict(Matrix.fromArray([[x / (2 * Math.PI)]])).get(0, 0) * 2 - 1;
  console.log(`sin(${x.toFixed(2)}) = ${actual.toFixed(3)}, predicted = ${predicted.toFixed(3)}`);
}
```

## What's Next?

This library goes far beyond basic networks:

- **Convolutional layers** — for image processing (`src/convolution.js`)
- **Recurrent networks** — for sequences (`src/rnn.js`, `src/lstm.js`, `src/gru.js`)
- **Transformers** — self-attention (`src/multi-head-attention.js`)
- **GANs** — generative models (`src/gan.js`)
- **Reinforcement learning** — Q-learning, policy gradients (`src/dqn.js`)
- **Autoencoders** — dimensionality reduction (`src/autoencoder.js`)

Check the `examples/` directory for more runnable demos, and `README.md` for the full feature list.

---

*Built from scratch in JavaScript. No dependencies. No magic — just math.*
