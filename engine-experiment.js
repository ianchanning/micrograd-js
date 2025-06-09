/**
 * WIP: This is an experiment to see if I can recreate karpathy's micrograd
 * {@link https://www.youtube.com/watch?v=VMj-3S1tku0}
 *
 * This is actually following along with his implementation and its not complete
 *
 * @see engine.js for the fully implemented AI version
 *
 * Note: This source uses more modern ES6 syntax
 * @param number data
 * @return object
 */
const value = (data, children = [], op = "") => {
  const _data = data;
  const _prev = children;
  // const _grad = 0;
  const _op = op;
  // can't use arrow function because we want to use `this`
  function add(other) {
    return value(_data + other.data, [this, other], "+");
  }
  function mul(other) {
    return value(_data * other.data, [this, other], "*");
  }
  return {
    add,
    mul,
    data: _data,
    prev: _prev,
    op: _op,
    toString: () => `Value(data=${_data})`,
  };
};

/**
 * Graph code
 */

/**
 * Function to generate the graph representation
 */
const buildGraph = (
  valueNode,
  graph = { nodes: [], edges: [] },
  idMap = new Map(),
) => {
  if (idMap.has(valueNode)) {
    return idMap.get(valueNode); // Return the existing node ID
  }
  const { op, prev } = valueNode;
  const { nodes, edges } = graph;

  // Store the current nodeId
  const nodeId = `node${idMap.size}`;
  idMap.set(valueNode, nodeId);

  // Add the current node
  const node = `${nodeId} [label="data=${valueNode.data}\\nop=${op}"];`;
  graph.nodes = [...nodes, node];

  // Add edges for children
  const childEdges = prev
    .map((child) => buildGraph(child, graph, idMap))
    .map((childId) => `${childId} -> ${nodeId};`);
  graph.edges = [...edges, childEdges];

  return nodeId;
};

/**
 * Function to render the graph
 */
const renderGraph = (rootValueNode) => {
  const graph = { nodes: [], edges: [] };
  buildGraph(rootValueNode, graph);

  const dot = `
    digraph G {
      ${graph.nodes.join("\n")}
      ${graph.edges.join("\n")}
    }
  `;

  const container = document.getElementById("root");
  container.innerHTML = Viz(dot);
};
const a = value(2.0);
const b = value(-3.0);
const c = value(10.0);
const d = a.mul(b).add(c);
console.log(a);
console.log(b);
console.log(c);
console.log(a.mul(b));
console.log(a.mul(b).add(c));

renderGraph(d);
