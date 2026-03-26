# ANE PWL Format Variations

Observed on M5 Pro (H17S), macOS 26.4, ANE compiler v9.509.

The ANE compiler generates different PWL configurations depending on function properties. This table documents the observed variation.

## Format Table

| Function | Entries | Header | Symmetric | Notes |
|----------|---------|--------|-----------|-------|
| sigmoid | 33 | [-1, 10, 0, 0.5] | Yes | Half-curve, offset=0.5 |
| tanh | 33 | [0, 4, 0, 1] | Yes | Half-curve, scale=1 |
| SiLU | 33 | [-10.2, inf, 0, inf] | No | Full range, 0.5 spacing |
| GELU | 33 | [-4.4, inf, 0, inf] | No | Narrower negative range |
| ELU | 32 | [-8, inf, -1, inf] | No | Clips at -1 for large negative |
| exp | 33 | [-25, 16, 0, inf] | No | Wide range |
| heaviside | 17 | [-1, 1, 0, 1] | No | Few points, narrow |
| leaky_relu | 0 | [-inf, inf, -inf, inf] | No | Slope params only |
| ceil/floor | 0 | [-inf, inf, 0, 0] | No | Different mechanism |

## Key Patterns

- **Symmetric bounded** (sigmoid, tanh): Header[3] = midpoint value. Hardware mirrors positive from negative.
- **Asymmetric unbounded** (SiLU, GELU, ELU): Header contains `inf` values. Full-range sampling.
- **Simple linear** (leaky_relu): No breakpoints. Slope parameters encode the function.
- **Conv pipeline**: Uses full-curve format with 32 entries at 0.5 spacing, replicated across 16 tiles.

## Why the Simulator Failed

We attempted to build a simulator that predicts the compiler's PWL format for arbitrary functions. The kill test showed that the compiler's format selection (entry count, spacing, range, symmetry mode) varies in ways we cannot predict from function properties alone.

The format is observable after compilation (by parsing .hwx binaries), but not predictable before compilation. This makes the simulator unreliable for functions outside the observed set.

## What Works Instead

The `ane_activation` tool sidesteps the format prediction problem entirely. Instead of predicting the compiler's PWL, it generates its own PWL via `torch.where` chains. The compiler then optimizes this to hardware-native representation. The error is known at generation time, before compilation.
