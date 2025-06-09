// engine.js - The liberated core of our tiny gradient beast!

/**
 * Our glorious pipe function, for composing operations in a readable, left-to-right flow.
 * @param  {...function} fns - A sequence of functions.
 * @returns {function} A new function that applies the sequence.
 */
export const pipe =
  (...fns) =>
  (x) =>
    fns.reduce((v, f) => f(v), x);

/**
 * Creates a Value object, the fundamental unit of our computation graph.
 * It's like a neuron, but simpler... and potentially more dangerous.
 * Private function, mostly. The outside world interacts via operations.
 *
 * @param {number} data - The raw numerical value.
 * @param {Value[]} [_children=[]] - Input Value objects that produced this one. Ancestry matters!
 * @param {string} [_op=''] - The operation that birthed this Value. Its 'tag'.
 * @param {string} [label=''] - A human-readable label, for debugging the madness.
 * @returns {Value} The freshly created Value object.
 */
function _createValue(data, _children = [], _op = "", label = "") {
  const v = {
    data: data,
    grad: 0, // Gradient initialized to zero. Innocent... for now.
    _prev: new Set(_children), // The parents, stored in a Set for uniqueness.
    _op: _op, // The operation symbol.
    label: label, // Its given name.
    // The Heretical Façade: a .backward() method for PyTorch API compatibility.
    // This will be defined as a closure below.
  };

  // This is the magic. The method is a closure that captures 'v' (the object itself).
  // It calls our pure core `backward` function, then performs the state update.
  v.backward = () => {
    const grads = backward(v); // Call the PURE core function.
    // Perform the "sin" of mutation deliberately and explicitly.
    for (const [node, grad] of grads.entries()) {
      node.grad = grad;
    }
  };

  return v;
}

/**
 * Creates a leaf Value node - represents an input or constant.
 * No parents, no operation. Pure data.
 *
 * @param {number} data - The initial data value.
 * @param {string} [label=''] - Optional label.
 * @returns {Value} A new leaf Value object.
 */
export function value(data, label = "") {
  return _createValue(data, [], "", label);
}

/**
 * Adds two Value objects. Elementary, Watson!
 *
 * @param {Value} v1 - First operand.
 * @param {Value} v2 - Second operand.
 * @param {string} [label=''] - Optional label for the resulting Value.
 * @returns {Value} A new Value object representing the sum.
 */
export function add(v1, v2, label = "") {
  // Ensure inputs are Value objects, basic sanity check
  if (
    !(v1 && typeof v1.data === "number") ||
    !(v2 && typeof v2.data === "number")
  ) {
    throw new Error(
      "Invalid inputs for add: Both inputs must be Value objects.",
    );
  }
  const out = _createValue(v1.data + v2.data, [v1, v2], "+", label);

  return out;
}

/**
 * Multiplies two Value objects. Let the chain rule shine!
 *
 * @param {Value} v1 - First operand.
 * @param {Value} v2 - Second operand.
 * @param {string} [label=''] - Optional label for the resulting Value.
 * @returns {Value} A new Value object representing the product.
 */
export function mul(v1, v2, label = "") {
  // Basic sanity check
  if (
    !(v1 && typeof v1.data === "number") ||
    !(v2 && typeof v2.data === "number")
  ) {
    throw new Error(
      "Invalid inputs for mul: Both inputs must be Value objects.",
    );
  }
  const out = _createValue(v1.data * v2.data, [v1, v2], "*", label);

  return out;
}

/**
 * Applies the hyperbolic tangent activation function to a Value object. Squishification!
 *
 * @param {Value} v - The input Value.
 * @param {string} [label=''] - Optional label.
 * @returns {Value} A new Value object representing tanh(v).
 */
export function tanh(v, label = "") {
  // Basic sanity check
  if (!(v && typeof v.data === "number")) {
    throw new Error("Invalid input for tanh: Input must be a Value object.");
  }
  const t = Math.tanh(v.data);
  const out = _createValue(t, [v], "tanh", label);

  return out;
}

/**
 * Raises a Value object to a power. For expressions that need a little... emphasis.
 *
 * @param {Value} v - The base Value.
 * @param {number} exponent - The exponent to raise the value to.
 * @param {string} [label=''] - Optional label.
 * @returns {Value} A new Value object representing v^exponent.
 */
export function pow(v, exponent, label = "") {
  if (!(v && typeof v.data === "number")) {
    throw new Error("Invalid input for pow: Input must be a Value object.");
  }
  const out = _createValue(v.data ** exponent, [v], `^${exponent}`, label);
  return out;
}

/**
 * Performs the backward pass (backpropagation) starting from a root node.
 * This is a PURE function. It takes a root node and returns a Map of
 * gradients, leaving the original graph untouched.
 *
 * @param {Value} root - The final Value object in the computation graph.
 * @returns {Map<Value, number>} A map from each node in the graph to its gradient.
 */
export function backward(root) {
  // Topological sort: Ensures we process nodes only after their dependents.
  const topo = [];
  const visited = new Set();
  function buildTopo(v) {
    if (v && !visited.has(v)) {
      visited.add(v);
      v._prev.forEach((child) => buildTopo(child));
      topo.push(v);
    }
  }

  if (!root) {
    console.error("Backward pass initiated on invalid root node.");
    return new Map();
  }
  buildTopo(root);

  // This map will store the gradients. No more mutating the Value objects!
  const grads = new Map();
  // Initialize all gradients to 0.
  visited.forEach((node) => grads.set(node, 0));
  // The gradient of the final output node with respect to itself is 1.
  grads.set(root, 1.0);

  // Iterate backwards through the sorted list to compute gradients.
  for (const node of topo.reverse()) {
    const nodeGrad = grads.get(node) || 0;

    // The corrected, centralized backprop logic, using a robust if/else if chain.
    const op = node._op;
    if (op === "+") {
      const [v1, v2] = [...node._prev];
      grads.set(v1, grads.get(v1) + nodeGrad);
      grads.set(v2, grads.get(v2) + nodeGrad);
    } else if (op === "*") {
      const [v1, v2] = [...node._prev];
      grads.set(v1, grads.get(v1) + v2.data * nodeGrad);
      grads.set(v2, grads.get(v2) + v1.data * nodeGrad);
    } else if (op === "tanh") {
      const [v] = [...node._prev];
      // The gradient can be calculated from the output value: 1 - out.data^2
      const tanhGrad = 1 - node.data * node.data;
      grads.set(v, grads.get(v) + tanhGrad * nodeGrad);
    } else if (op.startsWith("^")) {
      const [v] = [...node._prev];
      const exponent = Number(op.slice(1));
      const powGrad = exponent * v.data ** (exponent - 1);
      grads.set(v, grads.get(v) + powGrad * nodeGrad);
    }
    // Leaf nodes where op === "" are now correctly and implicitly ignored.
  }

  return grads;
}
