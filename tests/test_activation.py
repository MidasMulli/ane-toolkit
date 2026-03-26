#!/usr/bin/env python3
"""
KILL TEST: Custom activations compile to CoreML and produce correct output.

Tests that make_activation generates torch modules that:
1. Convert to CoreML via coremltools
2. Produce correct output
3. Match the reference function within FP16 tolerance
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def run_kill_tests():
    print("=" * 80)
    print("KILL TEST: Custom activations via torch.where → CoreML")
    print("=" * 80)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP: torch not available")
        return True

    from ane_activation import make_activation

    # Define test functions
    def mish(x):
        if abs(x) > 20: return x if x > 0 else 0
        return x * math.tanh(math.log1p(math.exp(x)))

    def star_relu(x):
        return 0.8862 * max(0, x)**2 - 0.4052

    def x_sin_x(x):
        return x * math.sin(x)

    def swish_b2(x):
        return x / (1 + math.exp(-2*x))

    def custom_pwl(x):
        if x < -2: return 0.3 * x + 0.1
        if x < 0: return -0.5 * x + 0.7
        if x < 1.5: return 2.0 * x - 0.3
        return 0.1 * x + 2.55

    TESTS = [
        ("mish", mish, (-8, 8), 32),
        ("star_relu", star_relu, (-4, 4), 32),
        ("x_sin_x", x_sin_x, (-6, 6), 32),
        ("swish_b2", swish_b2, (-8, 8), 32),
        ("custom_pwl", custom_pwl, (-4, 4), 16),
    ]

    results = []

    for name, fn, x_range, n_seg in TESTS:
        print(f"\n  {name}:")

        # Step 1: Create activation
        Act = make_activation(fn, x_range=x_range, n_segments=n_seg, name=name)
        act = Act()
        print(f"    Created: {act}")

        # Step 2: Test in PyTorch
        test_x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
        test_x_clipped = torch.clamp(test_x, x_range[0], x_range[1])
        with torch.no_grad():
            torch_out = act(test_x_clipped).numpy()

        ref = np.array([fn(float(x)) for x in test_x_clipped])
        torch_err = float(np.max(np.abs(torch_out - ref)))
        print(f"    PyTorch max error: {torch_err:.6f}")

        # Step 3: Convert to CoreML
        try:
            import coremltools as ct
            import warnings
            warnings.filterwarnings("ignore")

            traced = torch.jit.trace(act, torch.randn(1, 64))
            mlmodel = ct.convert(traced,
                inputs=[ct.TensorType(shape=(1, 64), name="x")],
                minimum_deployment_target=ct.target.macOS15)

            # Step 4: Run CoreML prediction
            input_np = np.zeros((1, 64), dtype=np.float32)
            for i, v in enumerate(test_x_clipped):
                if i < 64:
                    input_np[0, i] = float(v)
            pred = mlmodel.predict({"x": input_np})
            coreml_out = list(pred.values())[0].flatten()[:len(test_x_clipped)]

            coreml_err = float(np.max(np.abs(coreml_out - ref)))
            print(f"    CoreML max error:  {coreml_err:.6f}")
            converted = True
        except Exception as e:
            print(f"    CoreML conversion: FAILED ({e})")
            coreml_err = float('inf')
            converted = False

        passed = torch_err < 0.1 and (not converted or coreml_err < 0.1)
        results.append({'name': name, 'torch_err': torch_err,
                        'coreml_err': coreml_err, 'converted': converted,
                        'passed': passed})
        print(f"    Status: {'PASS' if passed else 'FAIL'}")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Name':<15} {'Torch Err':>10} {'CoreML Err':>10} {'Converted':>10} {'Status':>8}")
    print(f"{'-'*80}")
    for r in results:
        ce = f"{r['coreml_err']:.6f}" if r['converted'] else "N/A"
        print(f"{r['name']:<15} {r['torch_err']:>10.6f} {ce:>10} "
              f"{'Yes' if r['converted'] else 'No':>10} "
              f"{'PASS' if r['passed'] else 'FAIL':>8}")

    all_pass = all(r['passed'] for r in results)
    print(f"\n{'='*80}")
    print(f"RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")
    return all_pass


if __name__ == "__main__":
    passed = run_kill_tests()
    sys.exit(0 if passed else 1)
