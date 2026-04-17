// autograd.js — Reverse-Mode Automatic Differentiation
// Tape-based computation graph for automatic gradient computation
// Similar to PyTorch's autograd

// ===== Variable (tracked tensor) =====
export class Variable {
  constructor(value, { children = [], op = null, name = '' } = {}) {
    this.value = value;
    this.grad = 0;
    this.children = children; // [{variable, localGrad}]
    this.op = op;
    this.name = name;
    this._backward = null;
  }

  // Backward pass: compute gradients
  backward(gradOutput = 1) {
    // Topological sort
    const topo = [];
    const visited = new Set();
    const buildTopo = (v) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.children) {
        buildTopo(child.variable);
      }
      topo.push(v);
    };
    buildTopo(this);

    // Set gradient of output
    this.grad = gradOutput;

    // Reverse order
    for (let i = topo.length - 1; i >= 0; i--) {
      const node = topo[i];
      if (node._backward) node._backward();
    }
  }

  // Reset gradients
  zeroGrad() {
    this.grad = 0;
  }
}

// ===== Operations =====

export function add(a, b) {
  const out = new Variable(a.value + b.value, {
    children: [{ variable: a }, { variable: b }],
    op: 'add',
  });
  out._backward = () => {
    a.grad += out.grad;
    b.grad += out.grad;
  };
  return out;
}

export function mul(a, b) {
  const out = new Variable(a.value * b.value, {
    children: [{ variable: a }, { variable: b }],
    op: 'mul',
  });
  out._backward = () => {
    a.grad += b.value * out.grad;
    b.grad += a.value * out.grad;
  };
  return out;
}

export function sub(a, b) {
  const out = new Variable(a.value - b.value, {
    children: [{ variable: a }, { variable: b }],
    op: 'sub',
  });
  out._backward = () => {
    a.grad += out.grad;
    b.grad -= out.grad;
  };
  return out;
}

export function div(a, b) {
  const out = new Variable(a.value / b.value, {
    children: [{ variable: a }, { variable: b }],
    op: 'div',
  });
  out._backward = () => {
    a.grad += out.grad / b.value;
    b.grad -= out.grad * a.value / (b.value * b.value);
  };
  return out;
}

export function pow(a, n) {
  // n is a constant (not tracked)
  const out = new Variable(Math.pow(a.value, n), {
    children: [{ variable: a }],
    op: `pow(${n})`,
  });
  out._backward = () => {
    a.grad += n * Math.pow(a.value, n - 1) * out.grad;
  };
  return out;
}

export function neg(a) {
  const out = new Variable(-a.value, {
    children: [{ variable: a }],
    op: 'neg',
  });
  out._backward = () => {
    a.grad -= out.grad;
  };
  return out;
}

// ===== Activation Functions =====

export function relu(a) {
  const out = new Variable(Math.max(0, a.value), {
    children: [{ variable: a }],
    op: 'relu',
  });
  out._backward = () => {
    a.grad += (a.value > 0 ? 1 : 0) * out.grad;
  };
  return out;
}

export function sigmoid(a) {
  const s = 1 / (1 + Math.exp(-a.value));
  const out = new Variable(s, {
    children: [{ variable: a }],
    op: 'sigmoid',
  });
  out._backward = () => {
    a.grad += s * (1 - s) * out.grad;
  };
  return out;
}

export function tanh_ad(a) {
  const t = Math.tanh(a.value);
  const out = new Variable(t, {
    children: [{ variable: a }],
    op: 'tanh',
  });
  out._backward = () => {
    a.grad += (1 - t * t) * out.grad;
  };
  return out;
}

export function exp_ad(a) {
  const e = Math.exp(a.value);
  const out = new Variable(e, {
    children: [{ variable: a }],
    op: 'exp',
  });
  out._backward = () => {
    a.grad += e * out.grad;
  };
  return out;
}

export function log_ad(a) {
  const out = new Variable(Math.log(a.value), {
    children: [{ variable: a }],
    op: 'log',
  });
  out._backward = () => {
    a.grad += (1 / a.value) * out.grad;
  };
  return out;
}

export function sin_ad(a) {
  const out = new Variable(Math.sin(a.value), {
    children: [{ variable: a }],
    op: 'sin',
  });
  out._backward = () => {
    a.grad += Math.cos(a.value) * out.grad;
  };
  return out;
}

export function cos_ad(a) {
  const out = new Variable(Math.cos(a.value), {
    children: [{ variable: a }],
    op: 'cos',
  });
  out._backward = () => {
    a.grad -= Math.sin(a.value) * out.grad;
  };
  return out;
}

// ===== Convenience =====

export function constant(value) {
  return new Variable(value, { name: `const(${value})` });
}

export function parameter(value, name = '') {
  return new Variable(value, { name: name || `param(${value})` });
}

// Sum of array of Variables
export function sum(vars) {
  return vars.reduce((a, b) => add(a, b));
}

// Mean of array of Variables
export function mean(vars) {
  const s = sum(vars);
  const n = constant(vars.length);
  return div(s, n);
}

// MSE loss
export function mseLoss(predictions, targets) {
  const diffs = predictions.map((p, i) => {
    const pVar = p instanceof Variable ? p : constant(p);
    const t = targets[i] instanceof Variable ? targets[i] : constant(targets[i]);
    const diff = sub(pVar, t);
    return pow(diff, 2);
  });
  return mean(diffs);
}
