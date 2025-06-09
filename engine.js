// engine.js - The liberated core of our tiny gradient beast!

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
  return {
    data: data,
    grad: 0, // Gradient initialized to zero. Innocent... for now.
    _prev: new Set(_children), // The parents, stored in a Set for uniqueness.
    _op: _op, // The operation symbol.
    label: label, // Its given name.
  };
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
 * Performs the backward pass (backpropagation) starting from a root node.
 * This calculates the gradients for all nodes in the computation graph
 * leading to the root. It's where the learning happens!
 *
 * @param {Value} root - The final Value object in the computation graph. The 'output'.
 */
export function backward(root) {
  // Topological sort: Ensures we process nodes only after their dependents.
  // Crucial for correct gradient accumulation.
  const topo = [];
  const visited = new Set();
  function buildTopo(v) {
    if (v && !visited.has(v)) {
      // Check if v is valid before accessing properties
      visited.add(v);
      v._prev.forEach((child) => buildTopo(child));
      topo.push(v);
    }
  }
  buildTopo(root);

  // Initialize the gradient of the final output node to 1.
  // This signifies d(root)/d(root) = 1. The start of the chain rule.
  if (root && typeof root.grad !== "undefined") {
    // Check root is valid
    root.grad = 1.0;
  } else {
    console.error("Backward pass initiated on invalid root node:", root);
    return; // Stop if root is invalid
  }

  // Iterate backwards through the sorted list and call _backward for each node.
  // This section is now empty, awaiting the immutable gradient calculation logic.
}
