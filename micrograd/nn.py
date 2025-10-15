"""
Neural Network Module for micrograd
Implements basic building blocks: Neuron, Layer, and MLP (Multi-Layer Perceptron)
"""

import random
from micrograd.engine import Value

class Module:
    """
    Base class for all neural network modules.
    Provides common functionality like zeroing gradients and getting parameters.
    Similar to PyTorch's nn.Module
    """

    def zero_grad(self):
        """
        Reset all gradients to zero.
        This must be called before each backward pass to prevent gradient accumulation.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Return a list of all trainable parameters (weights and biases).
        Override this in subclasses.
        """
        return []

class Neuron(Module):
    """
    A single neuron (perceptron) with:
    - Multiple input connections (weights w)
    - One bias term (b)
    - Optional non-linear activation (ReLU)
    
    Formula: output = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
    """

    def __init__(self, nin, nonlin=True):
        """
        Initialize a neuron.
        
        Args:
            nin: Number of inputs (incoming connections)
            nonlin: If True, apply ReLU activation. If False, linear (no activation)
        
        Weights are randomly initialized between -1 and 1
        Bias is initialized to 0
        """
        # Create random weights for each input
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # Initialize bias to zero
        self.b = Value(0)
        # Store whether to use non-linear activation
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Forward pass: compute the neuron's output given inputs x.
        
        Args:
            x: List of input values [x1, x2, ..., xn]
        
        Returns:
            Value object representing the neuron's output
        
        Steps:
            1. Compute weighted sum: w1*x1 + w2*x2 + ... + wn*xn + b
            2. Apply activation function (ReLU or linear)
        """
        # Compute weighted sum of inputs plus bias
        # zip(self.w, x) pairs each weight with its input: [(w1,x1), (w2,x2), ...]
        # sum(..., self.b) adds the bias to the weighted sum
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        # ReLU(x) = max(0, x) introduces non-linearity
        # Linear means no activation (for output layer)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Return all trainable parameters of this neuron.
        
        Returns:
            List of Value objects: [w1, w2, ..., wn, b]
        """
        return self.w + [self.b]

    def __repr__(self):
        """String representation for printing."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """
    A layer of neurons (fully connected / dense layer).
    All neurons in the layer receive the same inputs.
    
    Example: Layer(2, 3) creates 3 neurons, each taking 2 inputs
             Input [x1, x2] → [neuron1, neuron2, neuron3] → [out1, out2, out3]
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Initialize a layer of neurons.
        
        Args:
            nin: Number of inputs to each neuron
            nout: Number of neurons in this layer (number of outputs)
            **kwargs: Additional arguments passed to Neuron (e.g., nonlin=False)
        """
        # Create nout neurons, each with nin inputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass: apply all neurons to the input.
        
        Args:
            x: Input values
        
        Returns:
            - Single Value if layer has 1 neuron (scalar output)
            - List of Values if layer has multiple neurons (vector output)
        """
        # Apply each neuron to the same input
        out = [n(x) for n in self.neurons]
        
        # Return scalar if single output, otherwise return list
        # This is for convenience when working with single-output layers
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Return all trainable parameters from all neurons in the layer.
        
        Returns:
            Flattened list of all weights and biases: [n1.w1, n1.w2, ..., n1.b, n2.w1, ...]
        """
        # List comprehension that flattens: for each neuron, get its parameters
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """String representation showing all neurons."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    Multi-Layer Perceptron: a sequence of layers forming a neural network.
    
    Example: MLP(2, [16, 16, 1]) creates:
        Input(2) → Layer(2→16) → Layer(16→16) → Layer(16→1) → Output(1)
    
    This is the complete neural network that chains layers together.
    """

    def __init__(self, nin, nouts):
        """
        Initialize a multi-layer perceptron.
        
        Args:
            nin: Number of input features
            nouts: List of output sizes for each layer
                   Example: [16, 16, 1] creates 3 layers with 16, 16, and 1 neurons
        
        Architecture:
            - All hidden layers use ReLU activation (nonlin=True)
            - Final layer uses linear activation (nonlin=False) for regression/scores
        """
        # sz contains the size of each layer: [nin, nouts[0], nouts[1], ...]
        # Example: MLP(2, [16, 16, 1]) → sz = [2, 16, 16, 1]
        sz = [nin] + nouts
        
        # Create layers by connecting consecutive sizes
        # Layer i connects sz[i] inputs to sz[i+1] outputs
        # All layers use ReLU except the last one (i != len(nouts)-1)
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Forward pass: pass input through all layers sequentially.
        
        Args:
            x: Input values (list of numbers or Value objects)
        
        Returns:
            Output of the final layer
        
        Data flows: x → layer1 → layer2 → ... → layerN → output
        """
        # Chain the layers: output of layer i becomes input of layer i+1
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Return all trainable parameters from all layers.
        
        Returns:
            Flattened list of all parameters in the entire network
        """
        # Flatten parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """String representation showing the complete network architecture."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"