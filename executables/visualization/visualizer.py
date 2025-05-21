#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import numpy as np
import matplotlib.pyplot as plt



width =     3000
height =    2000

data = np.fromfile("output/density.bin", dtype=np.float32)
data = data.reshape((height, width))

plt.imshow(data, cmap="inferno")
plt.colorbar(label="density")
plt.title("MBL density field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
