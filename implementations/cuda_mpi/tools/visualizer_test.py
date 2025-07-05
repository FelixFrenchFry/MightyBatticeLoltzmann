#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
FP = np.float64 if "--FP64" in sys.argv else np.float32

# simulation config
step =  25000
N_X =   10000
N_Y =   10000

# file path config
versionDirName =    "04"
subDirName =        "E"
dataType =          "velocity_x"

# path to raw .bin export:
def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

def get_file_path(data: str, step: int) -> str:
    return f"../../../exported/{versionDirName}/{subDirName}/{data}{format_step_suffix(step)}.bin"

# output dir
outputDirName = "output"
outputDir = f"{outputDirName}/{versionDirName}/test"
os.makedirs(outputDir, exist_ok=True)

# load and reshape data
bin_path = get_file_path(dataType, step)
A = np.fromfile(bin_path, dtype=FP).reshape((N_Y, N_X))

print(f"✅ Loaded {bin_path} | shape={A.shape} | min={A.min()} max={A.max()}")

# save plot
plt.figure(figsize=(8, 6))
plt.imshow(A, cmap="jet", origin="lower", aspect="auto")
plt.colorbar(label=dataType)
plt.title(f"{dataType} at step {step}")
plt.xlabel("X")
plt.ylabel("Y")

output_png = f"{outputDir}/{dataType}{format_step_suffix(step)}.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close()

print("min:", np.min(A))
print("max:", np.max(A))
print("mean:", np.mean(A))
print("contains NaN?", np.isnan(A).any())
print("contains inf?", np.isinf(A).any())

print(f"✅ Saved plot: {output_png}")
