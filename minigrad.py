"""
Minigrad: A tiny autograd engine for building and training neural networks.
Based on Andrej Karpathy's micrograd tutorial.
"""

import math
import random


# Define a Value object to store large datatypes for calculations
class Value:

    def __init__(self, data, _children=(), _op='', label=''):  # _children is an empty tuple, but we use it as a set for efficiency
        self.data = data
        self.grad = 0.0  # This is the gradient (slope) of a final Value object wrt its children or grandchildren (current node)
        self._backward = lambda: None  # Empty function, used to calculate chain rule for backpropagation
        self._prev = set(_children)  # This will be used for forward pass/backprop so the function knows the previous values used to reach this value
        self._op = _op  # _op is the operator represented using a string, used for forward pass/backprop 
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # Treat the other as a Value object it is already is not
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now."
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)    

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # Implementing topological sort to automatically do the _backward calls for backpropagation for self and each previous node in the toposort
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# Define a Neuron class for neural networks
class Neuron:

    def __init__(self, nin):    # nin: number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]    # Random weights between -1 and 1
        self.b = Value(random.uniform(-1, 1))    # Random bias between -1 and 1
        

    def __call__(self, x):
        # w*x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # sum(wi * xi) + b
        out = act.tanh()    # tanh(sum(wi * xi) + b)
        return out
    
    def parameters(self):
        return self.w + [self.b]


# Define a Layer class for the neurons
class Layer:

    def __init__(self, nin, nout):  # nout: number of outputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
    
# Define a multilayer perceptron (MLP) class
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Visualization utilities (optional - requires graphviz)
def trace(root):
    """Builds a set of all nodes and edges in a graph"""
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    """
    Visualizes the computation graph using graphviz.
    Requires: pip install graphviz
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("graphviz package required for visualization. Install with: pip install graphviz")
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # For any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data: %.4f | grad: %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # If this value is the result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # And connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


if __name__ == "__main__":
    # Example usage: Training a simple neural network
    print("Micrograd Example: Training a simple MLP\n")
    
    # Creating a tiny dataset
    xs = [
        [2.0, 3.0, -1.0, 6.0],
        [3.0, -1.0, 0.5, 2.0],
        [0.5, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -3.0]
    ]
    
    ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets
    
    # Define the MLP Neural Network
    model = MLP(4, [6, 4, 1])
    print(f"Size of Network: {len(model.parameters())} Parameters\n")
    
    # Training loop
    print("Training...")
    for k in range(50):
        # Forward pass
        ypred = [model(x) for x in xs]
        
        # Define the loss function (MSE loss)
        loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
        
        # Backward pass to set gradients
        # Flush the gradients to zero for backprop to prevent the gradients to accumulate 
        # (as we are doing self.grad += something in the Value object functions)
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        
        # Update the parameters using gradient descent
        for p in model.parameters():
            p.data += -0.05 * p.grad
        
        if k % 10 == 0:
            print(f"Step {k}, Loss: {loss.data:.6f}")
    
    print("\nFinal predictions:")
    ypred = [model(x) for x in xs]
    for i, (pred, target) in enumerate(zip(ypred, ys)):
        print(f"Input {i}: predicted={pred.data:.4f}, target={target}")