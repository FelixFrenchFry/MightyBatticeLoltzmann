#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
FP = np.float64 if "--FP64" in sys.argv else np.float32


# TODO: use quiverplot or streamplot?
# ----- VISUALIZATION OF THE VELOCITY VECTOR STREAMLINES -----
# simulation config
step =  1
N_X =   10000
N_Y =   10000
omega = 1.7
u_lid = 0.1

# output path config
dataType_A =        "velocity_x"
dataType_B =        "velocity_y"
outputDirName =     "output"
versionDirName =    "18"
subDirName =        "E"

def format_step_suffix(step: int, width: int = 9) -> str:
    return f"_{step:0{width}d}"

# data import file path
def get_file_path(data: str, step: int) -> str:
    return (
        f"../../../exported/"
        f"{versionDirName}/{subDirName}/{data}{format_step_suffix(step)}.bin")

# load velocity field
u_x = np.fromfile(get_file_path(dataType_A, step), dtype=FP).reshape((N_Y, N_X))
u_y = np.fromfile(get_file_path(dataType_B, step), dtype=FP).reshape((N_Y, N_X))

# downsample resolution
stride = 20
u_x_ds = u_x[::stride, ::stride]
u_y_ds = u_y[::stride, ::stride]

# create meshgrid for plotting
x = np.linspace(0, N_X, u_x_ds.shape[1], endpoint=False)
y = np.linspace(0, N_Y, u_x_ds.shape[0], endpoint=False)
X, Y = np.meshgrid(x, y)

# plot settings
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)
plt.figure(figsize=(6, 9))
speed = np.sqrt(u_x_ds**2 + u_y_ds**2)
plt.streamplot(X, Y, u_x_ds, u_y_ds, density=2.5, linewidth=1.25, arrowsize=1.0, color=speed, cmap='inferno')
plt.colorbar(label="Velocity magnitude")
plt.title(f"Streamlines at step {step}, omega {omega}, u_lid {u_lid}")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)

# save plot
outputPath = f"{outputDir}/streamlines{format_step_suffix(step)}.png"
plt.savefig(outputPath, dpi=300)
plt.close()
print(f"Saved streamplot: {outputPath}")
