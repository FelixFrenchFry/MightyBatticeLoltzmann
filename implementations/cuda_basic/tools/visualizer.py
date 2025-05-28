#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt



step =  1

# simulation config
N_X =   1500
N_Y =   1000

# output path config
dataType =          "velocity_magnitude"
outputDirName =     "output"
versionDirName =    "05"
subDirName =        "B"

def format_step_suffix(step: int, width: int = 6) -> str:
    return f"_{step:0{width}d}"

fileName = (
    f"../../../buildDir/implementations/cuda_basic/exported/"
    f"{versionDirName}/{subDirName}/{dataType}{format_step_suffix(step)}.bin"
)
data = np.fromfile(fileName, dtype=np.float32)
data = data.reshape((N_Y, N_X))

# plot settings
plt.imshow(data, origin='lower', cmap="seismic")
plt.colorbar(label="velocity x")
plt.title("MBL simulation step " + str(step))
plt.xlabel("X")
plt.ylabel("Y")

# output directory and file name settings
outputDir = f"{outputDirName}/{versionDirName}/{subDirName}"
os.makedirs(outputDir, exist_ok=True)
outputPath = f"{outputDir}/{dataType}{format_step_suffix(step)}.png"

plt.savefig(outputPath, dpi=300)
print(f"Saved plot: {outputPath}")
