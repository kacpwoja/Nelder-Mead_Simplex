import numpy as np

x = np.linspace(1, 10, 10)
y = np.linspace(11, 20, 10)
z = np.linspace(21, 30, 10)

a = np.array([x, y, z])

print(a)
a = np.transpose(a)
print(a[1])