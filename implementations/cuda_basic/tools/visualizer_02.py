#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32


# ----- VISUALIZATION OF THE X-VELOCITY DECAY (WAVE DECAY) -----
# simulation config
#steps = [1, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000,
#         225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000]
#steps = [1, 1000, 2000, 3000, 4000, 5000, 6000]
#steps = [1, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
steps = [1, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
N_X =   10000
N_Y =   15000

# output path config
dataType =          "velocity_x"
outputDirName =     "output"
versionDirName =    "13"
subDirName =        "L"

def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

# data import file path
def get_file_path(step: int) -> str:
    return (
        f"../../../exported/"
        f"{versionDirName}/{subDirName}/{dataType}{format_step_suffix(step)}.bin")

# load u_x along the y-axis at fixed x-position across the time steps
center_x = N_X // 2
profiles = []
for step in steps:
    data = np.fromfile(get_file_path(step), dtype=FP).reshape((N_Y, N_X))
    u_x_y = data[:, center_x]  # vertical slice at middle x value
    profiles.append(u_x_y)

# plot decay over time
plt.figure(figsize=(8, 8))
for i, profile in enumerate(profiles):
    label = f"Step {steps[i]}"
    plt.plot(profile, label=label)

# plot settings
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)
plt.title("Wave decay of X-velocity (u_x) along Y-axis")
plt.xlabel("Y")
plt.ylabel("u_x")
plt.legend(loc='upper right', fontsize=8)
plt.grid(True)

# save plot
outputPath = f"{outputDir}/wave_decay_x{format_step_suffix(steps[-1])}.png"
plt.savefig(outputPath, dpi=300)
plt.close()
print(f"Saved plot: {outputPath}")
