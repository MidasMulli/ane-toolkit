"""
Custom activation function generator for ANE deployment.

Converts any smooth function into a piecewise-linear PyTorch module
that compiles cleanly to CoreML MIL select/where ops, which the ANE
compiler optimizes into hardware PWL lookup tables.

The generated module:
- Works in PyTorch for training and evaluation
- Converts to CoreML via coremltools standard pipeline
- Compiles to ANE PWL table via the standard compiler
- No SIP, no binary patching, no root access required
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Type


class ANEActivation(nn.Module):
    """A piecewise-linear activation that compiles to ANE PWL tables.

    Forward pass computes the PWL approximation using torch.where chains.
    This converts to MIL select ops, which the ANE compiler recognizes
    and compiles into hardware PWL lookup tables.
    """

    def __init__(self, breakpoints_x: list, breakpoints_y: list, name: str = "custom"):
        super().__init__()
        self.name = name
        assert len(breakpoints_x) == len(breakpoints_y)
        assert len(breakpoints_x) >= 2

        # Store as buffers (not parameters — these are fixed, not trained)
        self.register_buffer('bp_x', torch.tensor(breakpoints_x, dtype=torch.float32))
        self.register_buffer('bp_y', torch.tensor(breakpoints_y, dtype=torch.float32))

        # Precompute slopes for each segment
        n = len(breakpoints_x)
        slopes = []
        intercepts = []
        for i in range(n - 1):
            dx = breakpoints_x[i + 1] - breakpoints_x[i]
            dy = breakpoints_y[i + 1] - breakpoints_y[i]
            slope = dy / dx if dx != 0 else 0.0
            intercept = breakpoints_y[i] - slope * breakpoints_x[i]
            slopes.append(slope)
            intercepts.append(intercept)

        # Extend: constant extrapolation outside range
        slopes.append(0.0)  # beyond last breakpoint
        intercepts.append(breakpoints_y[-1])

        self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))
        self.register_buffer('intercepts', torch.tensor(intercepts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build nested torch.where chain from right to left
        # This generates MIL select ops that compile to ANE
        n = len(self.bp_x)

        # Start with the rightmost segment (beyond last breakpoint)
        result = self.slopes[-1] * x + self.intercepts[-1]

        # Build from right to left
        for i in range(n - 2, -1, -1):
            segment = self.slopes[i] * x + self.intercepts[i]
            result = torch.where(x < self.bp_x[i + 1], segment, result)

        # Clamp below first breakpoint
        result = torch.where(x < self.bp_x[0],
                             self.slopes[0] * x + self.intercepts[0],
                             result)

        return result

    def extra_repr(self):
        return f"name={self.name}, segments={len(self.bp_x)-1}, range=[{self.bp_x[0]:.1f}, {self.bp_x[-1]:.1f}]"


def make_activation(
    fn: Callable[[float], float],
    x_range: Tuple[float, float] = (-8.0, 8.0),
    n_segments: int = 32,
    name: Optional[str] = None,
) -> Type[ANEActivation]:
    """Create an ANEActivation class from any smooth function.

    Samples the function at n_segments+1 evenly spaced points in FP16,
    then generates a piecewise-linear PyTorch module that compiles to
    ANE hardware PWL lookup tables via the standard CoreML pipeline.

    Args:
        fn: Any smooth function float -> float.
        x_range: Sampling range. Default (-8, 8) covers most activations.
        n_segments: Number of linear segments. 32 matches ANE hardware.
        name: Display name. Auto-detected if not provided.

    Returns:
        A callable that creates ANEActivation instances.

    Example:
        MyAct = make_activation(lambda x: x * math.tanh(math.softplus(x)))
        model = nn.Sequential(nn.Conv2d(64, 64, 3), MyAct())
    """
    if name is None:
        name = getattr(fn, '__name__', 'custom')

    x_min, x_max = x_range
    step = (x_max - x_min) / n_segments

    # Sample in FP16 (matching ANE precision)
    bp_x = []
    bp_y = []
    for i in range(n_segments + 1):
        x = x_min + i * step
        y = fn(x)
        bp_x.append(float(np.float16(x)))
        bp_y.append(float(np.float16(y)))

    # Create a factory that produces configured ANEActivation instances
    class ConfiguredActivation(ANEActivation):
        def __init__(self):
            super().__init__(bp_x, bp_y, name=name)

    ConfiguredActivation.__name__ = f"ANE_{name}"
    ConfiguredActivation.__qualname__ = f"ANE_{name}"

    # Quality check: compute max error and warn if high
    max_err = 0.0
    worst_x = 0.0
    n_test = 500
    for i in range(n_test):
        x = x_min + (x_max - x_min) * i / (n_test - 1)
        ref = fn(x)
        # PWL interpolation
        t = (x - x_min) / step
        k = int(t)
        k = min(k, n_segments - 1)
        frac = t - k
        y = bp_y[k] + frac * (bp_y[k + 1] - bp_y[k]) if k < n_segments else bp_y[-1]
        err = abs(y - ref)
        if err > max_err:
            max_err = err
            worst_x = x

    if max_err > 0.01:
        import sys
        print(f"Warning: {name} max approximation error is {max_err:.4f} at x={worst_x:.2f}.",
              file=sys.stderr)
        # Auto-test with doubled segments
        double_err = 0.0
        double_step = (x_max - x_min) / (n_segments * 2)
        double_y = [float(np.float16(fn(x_min + i * double_step))) for i in range(n_segments * 2 + 1)]
        for i in range(n_test):
            x = x_min + (x_max - x_min) * i / (n_test - 1)
            ref = fn(x)
            t = (x - x_min) / double_step
            k = int(t)
            k = min(k, n_segments * 2 - 1)
            frac = t - k
            y = double_y[k] + frac * (double_y[k + 1] - double_y[k]) if k < n_segments * 2 else double_y[-1]
            double_err = max(double_err, abs(y - ref))
        print(f"  With n_segments={n_segments*2}: max error would be {double_err:.4f}.",
              file=sys.stderr)
        if double_err < max_err * 0.6:
            print(f"  Consider: make_activation(..., n_segments={n_segments*2})",
                  file=sys.stderr)

    return ConfiguredActivation
