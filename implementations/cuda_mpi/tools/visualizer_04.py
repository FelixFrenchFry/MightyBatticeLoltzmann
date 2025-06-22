#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# TODO: use quiver plot or stream plot?
# ----- VISUALIZATION OF THE VELOCITY VECTOR STREAMLINES -----
# simulation config
N_X =   1000
N_Y =   1000
omega = 1.7
u_lid = 0.1

# step config
step_start =    1
step_end =      200000
step_stride =   50000
steps = [1] + list(range(step_stride, step_end + 1, step_stride))

# output path config
dataType_A =        "velocity_x"
dataType_B =        "velocity_y"
outputDirName =     "output"
versionDirName =    "03"
subDirName =        "C"

# formatting helper
def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

# data import file path
def get_file_path(data: str, step: int) -> str:
    return (
        f"../../../exported/"
        f"{versionDirName}/{subDirName}/{data}{format_step_suffix(step)}.bin")

# misc
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)
stride_plot = 20

for step in steps:
    try:
        # load velocity data
        u_x = np.fromfile(get_file_path(dataType_A, step), dtype=FP).reshape((N_Y, N_X))
        u_y = np.fromfile(get_file_path(dataType_B, step), dtype=FP).reshape((N_Y, N_X))

        # downsample data
        u_x_ds = u_x[::stride_plot, ::stride_plot]
        u_y_ds = u_y[::stride_plot, ::stride_plot]

        # create meshgrid
        x = np.linspace(0, N_X, u_x_ds.shape[1], endpoint=False)
        y = np.linspace(0, N_Y, u_x_ds.shape[0], endpoint=False)
        X, Y = np.meshgrid(x, y)

        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        speed = np.sqrt(u_x_ds**2 + u_y_ds**2)
        strm = ax.streamplot(
            X, Y, u_x_ds, u_y_ds,
            density=2.5, linewidth=1.25, arrowsize=1.0,
            color=speed, cmap='inferno')
        fig.colorbar(strm.lines, ax=ax, label="Velocity magnitude")
        ax.set_title(f"Streamlines at step {step}, omega {omega}, u_lid {u_lid}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(0, N_X)
        ax.set_ylim(0, N_Y)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.margins(0)

        # save as png
        outputPath = f"{outputDir}/streamlines{format_step_suffix(step)}.png"
        plt.savefig(outputPath, dpi=300)
        plt.close()
        print(f"✅  Saved plot: {outputPath}")

    except Exception as e:
        print(f"⚠️ Failed for step {step}: {e}")
