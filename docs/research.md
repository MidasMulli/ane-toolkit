# Research Background: H17 ANE Binary Format

This toolkit is built on reverse engineering of the Apple H17 ANE instruction set, conducted on M5 Pro (H17S) and M5 Air (H17G) running macOS 26.4.

## Key Findings

### Binary Format
The ANE executes Zin binaries (magic 0xBEEFFACE, CPU type 128). Two size classes exist:
- **48K** (49,152 bytes): Simple ops. 7 semantic words control the operation.
- **64K** (65,536 bytes): Complex ops with PWL lookup tables in `__KERN_0`.

### Conv Pipeline Activation Stage
Conv pipeline binaries have a configurable activation output stage at byte offset 0x4176:
- `0x10` = linear (no activation)
- `0x11` = relu
- `0x12` = PWL (reads lookup table from `__KERN_0`)

The PWL table is replicated 16 times at 640-byte stride (one per 16-channel ANE tile).

### Methodology
Systematic five-step process: compile → diff → map → mutate → verify. Applicable to any fixed-function accelerator with opaque binary format.

### What This Enabled
Understanding the PWL mechanism led to the discovery that `torch.where` piecewise-linear definitions compile through the standard CoreML pipeline to ANE PWL tables — enabling custom activations without binary patching or SIP.

Full format specification: see the [ane_reverse](https://github.com/MidasMulli/ane-toolkit) research directory.
