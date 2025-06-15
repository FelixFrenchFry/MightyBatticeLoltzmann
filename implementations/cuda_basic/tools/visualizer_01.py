#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# ----- VISUALIZATION OF THE X-VELOCITY -----
# simulation config
#steps = [1, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]
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

# find global u_min / u_max across all steps
all_data = []
for step in steps:
    data = np.fromfile(get_file_path(step), dtype=FP).reshape((N_Y, N_X))
    all_data.append(data)

all_data = np.stack(all_data)
u_min = np.min(all_data)
u_max = np.max(all_data)
print(f"Global color scale range: u_min={u_min:.6f} / u_max={u_max:.6f}\n")

# generate one image per step with shared color scale
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)

for i, step in enumerate(steps):
    data = all_data[i]

    # plot settings
    plt.figure(figsize=(8, 8))
    plt.imshow(data, origin='lower', cmap="seismic", vmin=u_min, vmax=u_max)
    plt.colorbar(label="velocity x")
    plt.title(f"X-velocity field (u_x) at {step}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # save plot
    outputPath = f"{outputDir}/velocity_x{format_step_suffix(step)}.png"
    plt.savefig(outputPath, dpi=300)
    plt.close()
    print(f"Saved plot: {outputPath}")
