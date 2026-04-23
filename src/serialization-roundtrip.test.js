// serialization-roundtrip.test.js — Round-trip serialization tests for all layer types
// For each layer type: create network, forward pass, serialize, deserialize, forward again,
// verify outputs match exactly (within floating point tolerance).

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Network } from './network.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';
import { Conv2D, MaxPool2D, Flatten } from './conv.js';
import { BatchNorm } from './batchnorm.js';
import { RNN, LSTM, GRU } from './rnn.js';
import { Dropout } from './dropout.js';
import { Embedding } from './embedding.js';
import { KANLayer } from './kan.js';
import { MixtureOfExperts } from './moe.js';

function assertMatrixClose(a, b, tol = 1e-10, msg = '') {
  assert.equal(a.rows, b.rows, `${msg} row mismatch`);
  assert.equal(a.cols, b.cols, `${msg} col mismatch`);
  for (let i = 0; i < a.data.length; i++) {
    assert.ok(Math.abs(a.data[i] - b.data[i]) < tol,
      `${msg} data[${i}] differs: ${a.data[i]} vs ${b.data[i]}`);
  }
}

describe('Serialization Round-trip: Dense', () => {
  it('round-trip preserves Dense network predictions', () => {
    const net = new Network();
    net.add(new Dense(3, 5, 'relu'));
    net.add(new Dense(5, 2, 'sigmoid'));

    const input = Matrix.random(1, 3);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'Dense round-trip');
  });

  it('save/load string round-trip', () => {
    const net = new Network();
    net.add(new Dense(4, 3, 'tanh'));
    net.loss('mse');

    const input = Matrix.random(1, 4);
    const out1 = net.forward(input);

    const str = net.save();
    const net2 = Network.fromJSON(str);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'Dense save/load');
  });
});

describe('Serialization Round-trip: Conv2D', () => {
  it('round-trip preserves Conv2D network predictions', () => {
    const net = new Network();
    net.layers.push(new Conv2D(8, 8, 1, 2, 3, 'relu'));
    net.layers.push(new Flatten());
    net.layers.push(new Dense(72, 2, 'sigmoid'));

    // Forward pass with random image (1 channel, 8x8)
    const input = Matrix.random(1, 64);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    
    const out2 = net2.forward(input);
    assertMatrixClose(out1, out2, 1e-10, 'Conv2D round-trip');
  });
});

describe('Serialization Round-trip: BatchNorm', () => {
  it('round-trip preserves BatchNorm network predictions', () => {
    const net = new Network();
    net.layers.push(new Dense(4, 8, 'linear'));
    const bn = new BatchNorm(8);
    bn.training = false; // Use running stats
    // Set some non-trivial running stats
    bn.runningMean = Matrix.random(1, 8);
    bn.runningVar = new Matrix(1, 8).map(() => Math.random() + 0.5); // positive variance
    net.layers.push(bn);
    net.layers.push(new Dense(8, 2, 'sigmoid'));

    const input = Matrix.random(1, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-8, 'BatchNorm round-trip');
  });
});

describe('Serialization Round-trip: RNN', () => {
  it('round-trip preserves RNN network predictions', () => {
    const net = new Network();
    net.layers.push(new RNN(4, 8));
    net.layers.push(new Dense(8, 2, 'sigmoid'));

    // RNN input: sequence of 3 timesteps, each with 4 features
    const input = Matrix.random(3, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'RNN round-trip');
  });
});

describe('Serialization Round-trip: LSTM', () => {
  it('round-trip preserves LSTM network predictions', () => {
    const net = new Network();
    net.layers.push(new LSTM(4, 6));
    net.layers.push(new Dense(6, 2, 'sigmoid'));

    const input = Matrix.random(3, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'LSTM round-trip');
  });
});

describe('Serialization Round-trip: GRU', () => {
  it('round-trip preserves GRU network predictions', () => {
    const net = new Network();
    net.layers.push(new GRU(4, 6));
    net.layers.push(new Dense(6, 2, 'sigmoid'));

    const input = Matrix.random(3, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'GRU round-trip');
  });
});

describe('Serialization Round-trip: Dropout', () => {
  it('round-trip preserves Dropout (inference mode) predictions', () => {
    const net = new Network();
    net.layers.push(new Dense(4, 8, 'relu'));
    const dropout = new Dropout(0.3);
    dropout.training = false; // Inference mode — dropout is a no-op
    net.layers.push(dropout);
    net.layers.push(new Dense(8, 2, 'sigmoid'));

    const input = Matrix.random(1, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'Dropout round-trip');
  });
});

describe('Serialization Round-trip: Embedding', () => {
  it('round-trip preserves Embedding weights', () => {
    const net = new Network();
    const emb = new Embedding(10, 4);
    net.layers.push(emb);

    // Embedding forward takes integer indices
    // Just verify the weights round-trip
    const json = net.toJSON();
    const net2 = Network.fromJSON(json);

    const emb2 = net2.layers[0];
    assert.equal(emb2.vocabSize, 10);
    assert.equal(emb2.embedDim, 4);
    assertMatrixClose(emb.weights, emb2.weights, 1e-10, 'Embedding weights');
  });
});

describe('Serialization Round-trip: KANLayer', () => {
  it('round-trip preserves KANLayer coefficients', () => {
    const kan = new KANLayer(3, 2, 5, 3);
    
    const net = new Network();
    net.layers.push(kan);

    const input = Matrix.random(1, 3);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'KAN round-trip');

    // Also verify coefficients match
    const kan2 = net2.layers[0];
    for (let i = 0; i < kan.coeffs.length; i++) {
      for (let j = 0; j < kan.coeffs[i].length; j++) {
        for (let k = 0; k < kan.coeffs[i][j].length; k++) {
          assert.equal(kan.coeffs[i][j][k], kan2.coeffs[i][j][k],
            `Coeff[${i}][${j}][${k}] mismatch`);
        }
      }
    }
  });
});

describe('Serialization Round-trip: MixtureOfExperts', () => {
  it('round-trip preserves MoE gate and expert weights', () => {
    const moe = new MixtureOfExperts(4, 8, 16, 4, 2);
    
    const net = new Network();
    net.layers.push(moe);

    const input = Matrix.random(1, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'MoE round-trip');
  });
});

describe('Serialization Round-trip: MaxPool2D', () => {
  it('round-trip preserves MaxPool2D config', () => {
    const net = new Network();
    net.layers.push(new Conv2D(8, 8, 1, 2, 3, 'relu'));
    net.layers.push(new MaxPool2D(6, 6, 2, 2));
    net.layers.push(new Flatten());
    net.layers.push(new Dense(18, 2, 'sigmoid'));

    const input = Matrix.random(1, 64);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'MaxPool2D round-trip');
  });
});

describe('Serialization: Complex Multi-Layer Networks', () => {
  it('Dense → BatchNorm → Dropout → Dense round-trip', () => {
    const net = new Network();
    net.layers.push(new Dense(4, 8, 'relu'));
    const bn = new BatchNorm(8);
    bn.training = false;
    bn.runningMean = Matrix.random(1, 8);
    bn.runningVar = new Matrix(1, 8).map(() => Math.random() + 0.5);
    net.layers.push(bn);
    const drop = new Dropout(0.5);
    drop.training = false;
    net.layers.push(drop);
    net.layers.push(new Dense(8, 2, 'sigmoid'));

    const input = Matrix.random(1, 4);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.fromJSON(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-8, 'Complex multi-layer round-trip');
  });

  it('serialization preserves loss function name', () => {
    const net = new Network();
    net.add(new Dense(4, 2, 'relu'));
    net.loss('cross_entropy');

    const json = net.toJSON();
    assert.ok(json.loss, 'Loss function name should be serialized');
    
    const net2 = Network.fromJSON(json);
    assert.ok(net2.lossFunction, 'Loss function should be restored');
  });

  it('Network.load() delegates to fromJSON correctly', () => {
    const net = new Network();
    net.add(new Dense(3, 2, 'relu'));
    
    const input = Matrix.random(1, 3);
    const out1 = net.forward(input);

    const json = net.toJSON();
    const net2 = Network.load(json);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'load() delegates correctly');
  });

  it('JSON string round-trip', () => {
    const net = new Network();
    net.add(new Dense(5, 3, 'tanh'));
    net.add(new Dense(3, 1, 'sigmoid'));
    
    const input = Matrix.random(1, 5);
    const out1 = net.forward(input);

    const str = JSON.stringify(net.toJSON());
    const net2 = Network.fromJSON(str);
    const out2 = net2.forward(input);

    assertMatrixClose(out1, out2, 1e-10, 'JSON string round-trip');
  });
});
