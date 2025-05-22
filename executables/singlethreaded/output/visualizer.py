#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt


N_X =   60
N_Y =   40

step =  1

fileName = f"../../../buildDir/executables/singlethreaded/velocity_magnitude_{step}.bin"
data = np.fromfile(fileName, dtype=np.float32)
data = data.reshape((N_Y, N_X))

plt.imshow(data, cmap="inferno")
plt.colorbar(label="velocity magnitudes")
plt.title("MBL simulation step " + str(step))
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("velocity_magnitudes_" + str(step) + ".png", dpi=300)
print("Saved plot to velocity_magnitudes_" + str(step) + ".png")
