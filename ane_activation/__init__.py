"""
ane_activation — Deploy custom activation functions to Apple Neural Engine.

Converts any Python function into a PyTorch module that compiles to ANE
via standard CoreML pipeline. No SIP required. No binary patching.

Usage:
    from ane_activation import make_activation
    import math

    MyAct = make_activation(
        lambda x: x * math.tanh(math.log(1 + math.exp(x))),  # Mish
        x_range=(-8, 8),
        n_segments=32,
    )

    class Model(nn.Module):
        def __init__(self):
            self.conv = nn.Conv2d(64, 64, 3)
            self.act = MyAct()
        def forward(self, x):
            return self.act(self.conv(x))
"""

from ane_activation.activation import make_activation, ANEActivation

__version__ = "0.1.0"
__all__ = ["make_activation", "ANEActivation"]
