#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt


N_X =   150
N_Y =   100

step =  0

fileName = f"../../../buildDir/implementations/cpu/01/output/velocity_x_{step}.bin"
data = np.fromfile(fileName, dtype=np.float32)
data = data.reshape((N_Y, N_X))

plt.imshow(data, origin='lower', cmap="seismic")
plt.colorbar(label="velocity x")
plt.title("MBL simulation step " + str(step))
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("velocity_x_" + str(step) + ".png", dpi=300)
print("Saved plot to velocity_x_" + str(step) + ".png")
