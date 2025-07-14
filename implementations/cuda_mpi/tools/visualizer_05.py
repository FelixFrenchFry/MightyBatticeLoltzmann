#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # fast backend
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# ----- VISUALIZATION OF THE X-VELOCITY AS HEATMAP -----
# simulation config
N_X =   12000
N_Y =   12000
omega = 1.2
u_lid = 0.1

# step config
step_start =    25_000
step_end =      12_000_000
step_stride =   25_000
steps = [1] + list(range(step_start, step_end + 1, step_stride))

# output path config
dataType =          "velocity_x"
outputDirName =     "output"
versionDirName =    "04"
subDirName =        "K"

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
stride_plot = 40

# plotting function for multiprocessing
def plot_step(step):
    try:
        # load velocity x-component
        u_x = load_velocity_component(get_file_path(dataType, step), FP)
        u_x_ds = u_x[::stride_plot, ::stride_plot]

        # plot as heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(u_x_ds, cmap='seismic', origin='lower',
                       extent=[0, N_X, 0, N_Y], aspect='equal',
                       vmin=-u_lid, vmax=u_lid)
        fig.colorbar(im, ax=ax, label="x-velocity (u_x)")
        #ax.set_title(f"y-velocity at step {step}, omega {omega}, u_lid {u_lid}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        ax.margins(0)

        # save as png
        outputPath = f"{outputDir}/{dataType}{format_step_suffix(step)}.png"
        plt.savefig(outputPath, dpi=300)
        plt.close()

        # free up memory
        del u_x, u_x_ds, im, fig, ax
        import gc; gc.collect()

        print(f"✅  Saved plot: {outputPath}")

    except Exception as e:
        print(f"⚠️ Failed for step {step}: {e}")

if __name__ == "__main__":
    with mp.Pool(10) as pool:
        pool.map(plot_step, steps)
