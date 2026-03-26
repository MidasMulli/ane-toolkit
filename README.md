# ane-toolkit

Deploy custom activation functions to Apple Neural Engine. No SIP. No binary patching. Standard CoreML pipeline.

## What This Does

Converts any Python function into a PyTorch module that compiles to ANE hardware via coremltools. The generated module uses `torch.where` chains that the CoreML compiler recognizes and optimizes into hardware piecewise-linear (PWL) lookup tables.

```python
from ane_activation import make_activation
import math

# Any smooth function → ANE-deployable activation
Mish = make_activation(
    lambda x: x * math.tanh(math.log(1 + math.exp(x))),
    x_range=(-8, 8), n_segments=32
)

# Drop into your model
class MyModel(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(64, 64, 3, padding=1)
        self.act = Mish()
    def forward(self, x):
        return self.act(self.conv(x))
```

Converts to CoreML with standard `coremltools.convert()`. Runs on ANE on every Apple Silicon device. No special permissions required.

## Why This Exists

The ANE compiler fuses standard activations (ReLU, SiLU, GELU) into conv pipelines automatically. But if your activation isn't in CoreML's built-in set, the compiler can't optimize it and may fall back to CPU.

This toolkit bridges the gap: define any activation as a mathematical function, and it generates the PWL approximation that the ANE hardware can execute natively.

## Tested Activations

| Activation | Definition | Max Error | Segments |
|-----------|-----------|-----------|----------|
| Mish | `x·tanh(ln(1+eˣ))` | 0.000377 | 32 |
| Swish-β2 | `x·σ(2x)` | 0.000395 | 32 |
| StarReLU | `0.886·relu(x)² - 0.405` | 0.001025 | 32 |
| Custom PWL | 4-segment piecewise linear | 0.000391 | 16 |
| x·sin(x) | `x·sin(x)` | 0.042228 | 32 |

All tested end-to-end: Python function → PyTorch → CoreML → ANE execution.

Error is measured against the exact function evaluated in FP64. The dominant error source is FP16 quantization and piecewise-linear interpolation between breakpoints.

## Install

```bash
pip install numpy torch coremltools
# Then copy ane_activation/ to your project, or:
pip install -e .
```

Requires: Python 3.11+, PyTorch, coremltools. macOS 15+ for CoreML deployment.

## API

### `make_activation(fn, x_range, n_segments, name)`

Creates a PyTorch `nn.Module` class from any callable.

```python
from ane_activation import make_activation

# Simple: lambda
ReLU6 = make_activation(lambda x: min(max(x, 0), 6), x_range=(0, 6), n_segments=16)

# Complex: any smooth function
def my_gating(x):
    return x * (1 / (1 + math.exp(-1.702 * x)))  # GELU approximation
GatedAct = make_activation(my_gating, x_range=(-6, 6), n_segments=32)

# Use in model
model = nn.Sequential(nn.Linear(256, 256), GatedAct())
```

**Parameters:**
- `fn`: Any `float → float` callable
- `x_range`: Tuple `(min, max)` — values outside this range are clamped
- `n_segments`: Number of linear segments (default 32, matches ANE hardware)
- `name`: Display name for the module

**Returns:** A class (not instance) that creates `ANEActivation` modules.

### Error Control

More segments = lower error but more `torch.where` ops in the MIL graph:

| Segments | Typical Max Error | MIL Ops |
|----------|------------------|---------|
| 8 | ~0.05 | ~25 |
| 16 | ~0.01 | ~50 |
| 32 | ~0.001 | ~100 |
| 64 | ~0.0003 | ~200 |

32 segments is the sweet spot — matches the hardware PWL table size and keeps MIL graph manageable.

## How It Works

1. **Sampling**: The function is evaluated at `n_segments + 1` evenly spaced points, rounded to FP16 (matching ANE precision).

2. **PWL Generation**: A `torch.where` chain encodes the piecewise-linear approximation. Each segment is `slope * x + intercept` between consecutive breakpoints.

3. **CoreML Conversion**: `coremltools.convert()` maps `torch.where` to MIL `select` ops.

4. **ANE Compilation**: The CoreML compiler recognizes the piecewise-linear structure and compiles it to hardware PWL lookup tables in the ANE's conv pipeline output stage — the same mechanism used for built-in activations like sigmoid and tanh.

## Limitations

- **Approximation, not exact**: The output is a piecewise-linear approximation. For smooth functions with low curvature, 32 segments typically achieves < 0.001 max error. For rapidly varying functions (e.g., `x·sin(x)` with many oscillations), error increases.

- **FP16 precision**: All ANE computation is FP16. Values outside the ~[-65504, 65504] range overflow.

- **Clamped outside range**: Input values below `x_range[0]` or above `x_range[1]` are extrapolated linearly from the nearest segment, not the true function.

- **Compilation may vary**: The CoreML compiler's optimization is a black box. The `torch.where` chain should compile to ANE PWL tables, but Apple may change compiler behavior across macOS versions.

- **Not verified on all hardware**: Tested on M5 Pro (macOS 26.4). The approach uses standard CoreML APIs and should work on all Apple Silicon, but ANE-specific behavior may vary.

## Research Background

This toolkit is built on reverse engineering of the Apple H17 ANE binary format. The research documented the hardware PWL lookup table mechanism that the ANE uses for activation functions, which informed the design of this tool.

Key findings:
- The ANE evaluates activations via 32-point FP16 piecewise-linear lookup tables
- The compiler generates these tables from MIL operations including `select` (torch.where)
- Conv + activation fuse into a single hardware dispatch at zero additional cost
- The `torch.where` → MIL `select` → ANE PWL path works without SIP or binary patching

See [docs/binary_format.md](docs/binary_format.md) for the full H17 ANE binary format specification. See [docs/methodology.md](docs/methodology.md) for the reverse engineering methodology.

## Prior Art

- **geohot** (2020): First public HWX format analysis, BEEFFACE magic identification. [tinygrad](https://github.com/tinygrad/tinygrad)
- **freedomtan** (2021): `coreml_to_ane_hwx` binary extraction tools. [GitHub](https://github.com/nicklolsen/coreml_to_ane_hwx)
- **hollance** (2022): CoreML performance characterization, ANE tensor layout. [GitHub](https://github.com/hollance)
- **eiln** (2023-2024): Linux ANE driver RE, IOKit mapping, H13/H14 analysis. [GitHub](https://github.com/eiln/ane)
- **maderix** (2025): `_ANEClient` API, IOSurface format, direct evaluation. [GitHub](https://github.com/maderix/apple-ane-exploration)
- **Orion** (2025-2026): `_ANEInMemoryModel` MIL compilation, espresso.net mode patching.

The conv pipeline activation byte (0x4176), tile-replicated PWL format, 48K word map, mode sweep, and the `torch.where` → ANE PWL deployment path are original contributions.

## License

MIT
