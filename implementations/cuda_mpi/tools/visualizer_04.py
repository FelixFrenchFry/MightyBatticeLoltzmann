#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # fast backend
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.colors as mcolors
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# TODO: use quiver plot or stream plot?
# ----- VISUALIZATION OF THE VELOCITY AS VECTOR STREAMLINES -----
# simulation config
N_X =   24000
N_Y =   24000
omega = 1.1
u_lid = 0.1

# step config
step_start =    100_000
step_end =      56_400_000
step_stride =   100_000
steps = [1] + list(range(step_start, step_end + 1, step_stride))

# output path config
dataType_A =        "velocity_x"
dataType_B =        "velocity_y"
outputDirName =     "output"
versionDirName =    "04"
subDirName =        "M_FINAL"

# formatting helper
def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

# data import file path
def get_file_path(data: str, step: int) -> str:
    return (
        f"../../../exported/"
        f"{versionDirName}/{subDirName}/{data}{format_step_suffix(step)}.bin")

# lazy loading
def load_velocity_component(filename: str, dtype: np.dtype) -> np.ndarray:
    return np.memmap(filename, dtype=dtype, mode='r', shape=(N_Y, N_X))

# misc
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)
stride_plot = 20

# font sizes
font_axes = 16
font_titles = 16
font_legend = 16

# single step processing function
def plot_step(step: int):
    try:
        u_x = load_velocity_component(get_file_path(dataType_A, step), FP)
        u_y = load_velocity_component(get_file_path(dataType_B, step), FP)

        u_x_ds = u_x[::stride_plot, ::stride_plot]
        u_y_ds = u_y[::stride_plot, ::stride_plot]

        x = np.linspace(0, N_X, u_x_ds.shape[1], endpoint=False)
        y = np.linspace(0, N_Y, u_x_ds.shape[0], endpoint=False)
        X, Y = np.meshgrid(x, y)

        speed = np.sqrt(u_x_ds**2 + u_y_ds**2)

        fig, ax = plt.subplots(figsize=(6, 5))
        norm = mcolors.Normalize(vmin=0.0, vmax=u_lid)
        strm = ax.streamplot(
            X, Y, u_x_ds, u_y_ds,
            density=1.5, linewidth=1.5, arrowsize=1.0,
            color=speed, cmap='inferno', norm=norm)
        fig.colorbar(strm.lines, ax=ax, pad=0.1)
        ax.set_title(f"{step:>9,}", fontsize=font_titles)
        #ax.set_xlabel("X", fontsize=font_axes)
        #ax.set_ylabel("Y", fontsize=font_axes)
        ax.set_xlim(0, N_X)
        ax.set_ylim(0, N_Y)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.margins(0)

        outputPath = f"{outputDir}/streamlines{format_step_suffix(step)}.png"
        plt.savefig(outputPath, dpi=300)
        plt.close()

        del u_x, u_y, u_x_ds, u_y_ds, X, Y, speed, strm, fig, ax
        import gc; gc.collect()

        print(f"✅  Saved plot: {outputPath}")

    except Exception as e:
        print(f"⚠️ Failed for step {step}: {e}")

# run with multiprocessing
if __name__ == "__main__":
    with mp.Pool(10) as pool:
        pool.map(plot_step, steps)
