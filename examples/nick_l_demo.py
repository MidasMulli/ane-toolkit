#!/usr/bin/env python3
"""
ane-toolkit demo: custom activation on Apple Neural Engine.

Encodes "Nick L 3-26-2026" as negative ASCII values, runs through
a custom |x| activation (not in CoreML's built-in set), decodes
the output back to text.

3 lines to define the activation. Standard CoreML pipeline. No SIP.
"""

import math
from ane_activation import make_activation

# 1. Define custom activation
AbsAct = make_activation(lambda x: abs(x), x_range=(-128, 128), n_segments=32)

# 2. Encode message as negative ASCII
message = "Nick L 3-26-2026"
encoded = [-ord(c) for c in message]

# 3. Run through PyTorch
import torch
act = AbsAct()
with torch.no_grad():
    output = act(torch.tensor([encoded], dtype=torch.float32))
decoded = ''.join(chr(int(round(v))) for v in output[0])

print(f'Input:   {encoded}')
print(f'Output:  {[int(round(v)) for v in output[0]]}')
print(f'Decoded: "{decoded}"')
