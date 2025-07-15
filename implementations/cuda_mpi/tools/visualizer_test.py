#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # fast backend
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32



# ----- VISUALIZATION OF THE X/Y-VELOCITY AS A SHARED HEATMAP -----
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
dataType_A =        "velocity_x"
dataType_B =        "velocity_y"
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

# font sizes
font_axes = 16
font_titles = 20
font_legend = 16

# plotting function for multiprocessing
def plot_step(step):
    try:
        # load x and y velocity components
        u_x = load_velocity_component(get_file_path("velocity_x", step), FP)
        u_y = load_velocity_component(get_file_path("velocity_y", step), FP)
        u_x_ds = u_x[::stride_plot, ::stride_plot]
        u_y_ds = u_y[::stride_plot, ::stride_plot]

        # create shared figure
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), constrained_layout=True)

        # plot u_x
        im1 = ax1.imshow(u_x_ds, cmap='seismic', origin='lower',
                         extent=[0, N_X, 0, N_Y], aspect='equal',
                         vmin=-u_lid, vmax=u_lid)
        #ax1.set_title(f"x-velocity (u_x)", fontsize=font_titles)
        #ax1.set_xlabel("X", fontsize=font_axes)
        #ax1.set_ylabel("Y", fontsize=font_axes)
        ax1.grid(False)
        ax1.margins(0)

        # plot u_y
        im2 = ax2.imshow(u_y_ds, cmap='seismic', origin='lower',
                         extent=[0, N_X, 0, N_Y], aspect='equal',
                         vmin=-u_lid, vmax=u_lid)
        #ax2.set_title(f"y-velocity (u_y)", fontsize=font_titles)
        #ax2.set_xlabel("X", fontsize=font_axes)
        #ax2.set_ylabel("Y", fontsize=font_axes)
        ax2.grid(False)
        ax2.margins(0)

        # add one shared colorbar in the center
        cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', pad=0.05)
        #cbar.set_label("velocity", fontsize=font_legend)
        cbar.ax.tick_params(labelsize=10)

        # save as png
        outputPath = f"{outputDir}/velocity_shared{format_step_suffix(step)}.png"
        plt.savefig(outputPath, dpi=300)
        plt.close()

        # free up memory
        del u_x, u_y, u_x_ds, u_y_ds, im1, im2, fig, ax1, ax2
        import gc; gc.collect()

        print(f"✅  Saved plot: {outputPath}")

    except Exception as e:
        print(f"⚠️ Failed for step {step}: {e}")

if __name__ == "__main__":
    with mp.Pool(10) as pool:
        pool.map(plot_step, steps)
