"""
Autograd Engine for micrograd
This implements automatic differentiation (backpropagation) for scalar values.

Key concept: Every operation creates a computation graph that tracks how values
depend on each other. We can then compute gradients automatically using the chain rule.
"""

class Value:
    """
    Wraps a scalar value and tracks operations for automatic differentiation.
    
    The Value class is the core of micrograd's autograd engine. It:
    1. Stores a scalar data value
    2. Stores the gradient (derivative) with respect to some loss
    3. Tracks the computational graph (which operations created this value)
    4. Implements backpropagation to compute gradients
    
    Example:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b  # c.data = 6.0
        c.backward()  # Compute gradients
        print(a.grad)  # Gradient of c with respect to a (= 3.0)
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a Value object.
        
        Args:
            data: The actual numerical value (float or int)
            _children: Tuple of Value objects that were used to create this value
            _op: String describing the operation ('+', '*', 'ReLU', etc.)
        
        Attributes:
            self.data: The scalar value
            self.grad: Gradient (derivative) initialized to 0
            self._backward: Function to propagate gradients backward
            self._prev: Set of parent Values in the computation graph
            self._op: Operation that created this Value (for visualization)
        """
        self.data = data
        self.grad = 0  # Gradient starts at 0, computed during backward pass
        
        # Internal variables used for autograd graph construction
        self._backward = lambda: None  # Default: no gradient computation
        self._prev = set(_children)     # Parents in computation graph
        self._op = _op                  # Operation label for debugging/visualization

    def __add__(self, other):
        """
        Addition operation with automatic differentiation.
        
        Forward: out = self + other
        Backward: d(out)/d(self) = 1, d(out)/d(other) = 1
        
        Example:
            a = Value(2.0)
            b = Value(3.0)
            c = a + b  # c.data = 5.0
            c.backward()
            # a.grad = 1.0 (derivative of c w.r.t. a)
            # b.grad = 1.0 (derivative of c w.r.t. b)
        """
        # Convert other to Value if it's a plain number
        other = other if isinstance(other, Value) else Value(other)
        
        # Create output Value with sum
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """
            Backpropagate gradients through addition.
            
            Chain rule: if out = self + other, then:
                d(Loss)/d(self) = d(Loss)/d(out) * d(out)/d(self)
                               = out.grad * 1
            
            += is used because a variable might be used multiple times
            (gradient accumulation)
            """
            self.grad += out.grad   # Gradient flows equally to both inputs
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiplication operation with automatic differentiation.
        
        Forward: out = self * other
        Backward: d(out)/d(self) = other, d(out)/d(other) = self
        
        Example:
            a = Value(2.0)
            b = Value(3.0)
            c = a * b  # c.data = 6.0
            c.backward()
            # a.grad = 3.0 (derivative of c w.r.t. a = b.data)
            # b.grad = 2.0 (derivative of c w.r.t. b = a.data)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            Backpropagate gradients through multiplication.
            
            Chain rule: if out = self * other, then:
                d(Loss)/d(self) = d(Loss)/d(out) * d(out)/d(self)
                               = out.grad * other.data
            
            This is the product rule from calculus.
            """
            self.grad += other.data * out.grad  # d(a*b)/da = b
            other.grad += self.data * out.grad  # d(a*b)/db = a
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Power operation with automatic differentiation.
        
        Forward: out = self ** other
        Backward: d(out)/d(self) = other * self^(other-1)
        
        Note: Only supports constant exponents (int/float), not Value exponents.
        
        Example:
            a = Value(3.0)
            b = a ** 2  # b.data = 9.0
            b.backward()
            # a.grad = 6.0 (derivative = 2 * 3^1 = 6)
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            """
            Backpropagate gradients through power operation.
            
            Power rule: d(x^n)/dx = n * x^(n-1)
            """
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        ReLU (Rectified Linear Unit) activation function.
        
        Forward: out = max(0, self)
        Backward: d(out)/d(self) = 1 if self > 0, else 0
        
        ReLU is the most common activation function in neural networks.
        It introduces non-linearity while being simple to compute.
        
        Example:
            a = Value(-2.0)
            b = a.relu()  # b.data = 0.0 (negative values become 0)
            
            c = Value(3.0)
            d = c.relu()  # d.data = 3.0 (positive values unchanged)
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            """
            Backpropagate gradients through ReLU.
            
            Gradient is:
                - 1 if input was positive (gradient flows through)
                - 0 if input was negative (gradient is blocked)
            
            This is why ReLU helps with the "vanishing gradient" problem.
            """
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Compute gradients for all Values in the computation graph.
        
        This is the core of backpropagation. It:
        1. Builds a topological ordering of the computation graph
        2. Initializes the gradient of the output to 1 (d(out)/d(out) = 1)
        3. Applies the chain rule in reverse topological order
        
        The topological order ensures we compute gradients for a node only
        after computing gradients for all nodes that depend on it.
        
        Example:
            a = Value(2.0)
            b = Value(3.0)
            c = a * b
            d = c + 1
            d.backward()  # Computes gradients for a, b, c, d
            print(a.grad)  # 3.0
            print(b.grad)  # 2.0
        """
        # Build topological ordering of all nodes in the computation graph
        # Topological sort ensures we visit nodes in the correct order for backprop
        topo = []
        visited = set()
        
        def build_topo(v):
            """
            Recursively build topological ordering using depth-first search.
            
            A node is added to topo only after all its children are added.
            This ensures we process gradients from outputs to inputs.
            """
            if v not in visited:
                visited.add(v)
                # Recursively visit all children (inputs to this operation)
                for child in v._prev:
                    build_topo(child)
                # Add current node after all children
                topo.append(v)
        
        build_topo(self)

        # Apply chain rule to compute gradients
        # Start with gradient of 1 for the output (this node)
        self.grad = 1
        
        # Go through nodes in reverse topological order
        # This ensures we've computed the gradient of all nodes that depend on v
        # before we compute v's gradient
        for v in reversed(topo):
            v._backward()  # Call the backward function for this operation

    # Convenience methods to make Value behave like regular numbers
    # These enable natural mathematical expressions with Value objects

    def __neg__(self):
        """Unary negation: -self"""
        return self * -1

    def __radd__(self, other):
        """Reverse addition: other + self (when other is not a Value)"""
        return self + other

    def __sub__(self, other):
        """Subtraction: self - other"""
        return self + (-other)

    def __rsub__(self, other):
        """Reverse subtraction: other - self"""
        return other + (-self)

    def __rmul__(self, other):
        """Reverse multiplication: other * self"""
        return self * other

    def __truediv__(self, other):
        """Division: self / other (implemented as self * other^-1)"""
        return self * other**-1

    def __rtruediv__(self, other):
        """Reverse division: other / self"""
        return other * self**-1

    def __repr__(self):
        """String representation for printing/debugging."""
        return f"Value(data={self.data}, grad={self.grad})"


"""
Key Insights:

1. Computational Graph: Every operation creates nodes and edges in a directed
   acyclic graph (DAG). This graph represents how the output depends on inputs.

2. Forward Pass: Computing the output value (self.data) by applying operations.

3. Backward Pass: Computing gradients (self.grad) using the chain rule in
   reverse topological order.

4. Chain Rule: If z = f(y) and y = g(x), then dz/dx = (dz/dy) * (dy/dx)
   This is why we multiply out.grad by the local gradient in _backward().

5. Gradient Accumulation: We use += for gradients because a variable might
   be used multiple times (e.g., x + x). Each use contributes to the total gradient.

Example of complete flow:
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b        # c = -6.0, tracks that c depends on a and b
    d = c.relu()     # d = 0.0, tracks that d depends on c
    d.backward()     # Computes all gradients
    
    # Gradients:
    # d.grad = 1.0 (always starts at 1)
    # c.grad = 0.0 (ReLU blocks gradient when input < 0)
    # a.grad = 0.0 (no gradient flows through)
    # b.grad = 0.0 (no gradient flows through)
"""