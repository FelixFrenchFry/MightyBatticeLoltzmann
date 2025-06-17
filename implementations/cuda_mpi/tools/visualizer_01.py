#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# ----- VISUALIZATION OF THE X-VELOCITY -----
# simulation config
N_X =   60
N_Y =   40
omega = 1.7

# step config
step_start =    1
step_end =      200
step_stride =   10
steps = [1] + list(range(step_stride, step_end + 1, step_stride))

# output path config
dataType =          "velocity_x"
outputDirName =     "output"
versionDirName =    "01"
subDirName =        "A"

# formatting helper
def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

# data import file path
def get_file_path(step: int) -> str:
    return f"../../../exported/{versionDirName}/{subDirName}/{dataType}{format_step_suffix(step)}.bin"

# misc
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)

# determine global color scale
u_min, u_max = None, None
for step in steps:
    data = np.fromfile(get_file_path(step), dtype=FP).reshape((N_Y, N_X))
    step_min, step_max = np.min(data), np.max(data)
    if u_min is None or step_min < u_min:
        u_min = step_min
    if u_max is None or step_max > u_max:
        u_max = step_max

#print(f"\n✅ Global color scale: u_min={u_min:.6f}, u_max={u_max:.6f}\n")

# generate plots
for step in steps:
    data = np.fromfile(get_file_path(step), dtype=FP).reshape((N_Y, N_X))

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(data, origin='lower', cmap="seismic", vmin=u_min, vmax=u_max, extent=[0, N_X, 0, N_Y])

    cbar = fig.colorbar(img, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label("Velocity X")

    ax.set_title(f"X-velocity field (u_x) at step {step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_xlim(0, N_X)
    ax.set_ylim(0, N_Y)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.margins(0)

    outputPath = f"{outputDir}/{dataType}{format_step_suffix(step)}.png"
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅  Saved plot: {outputPath}")
