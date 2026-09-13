"""
Ignition Quickstart: Sub-100 microsecond ONNX inference on AMD Phoenix XDNA1 NPU.
"""
import numpy as np
import ignition

# 1. Compile ONNX model for physical AMD Phoenix AIE2 NPU silicon
model = ignition.compile("model.onnx", backend="xdna1")

# 2. Execute inference with unswizzled spatial output
input_activation = np.zeros((1, 8, 32, 32), dtype=np.int8)
output = model.predict(input_activation)
print(f"[+] Output tensor shape: {output.shape}, sample: {output.flatten()[:4]}")

# 3. Profile sustained physical hardware throughput (0 intermediate DDR bytes)
report = model.benchmark(iterations=200)
print(f"[+] Sustained Throughput: {report.fps:.1f} FPS | Mean Latency: {report.mean_us:.2f} μs")
