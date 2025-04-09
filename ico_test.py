from icosphere import icosphere
import scipy.spatial as sp
import numpy as np

a_sphere = 20
goal = 1 / a_sphere

N = np.array([])
seps = np.array([])
for i in range(0, 21):
    ico_nodes, _ = icosphere(i)
    n = ico_nodes.shape[0]
    sep = np.min(sp.distance.pdist(ico_nodes))
    seps = np.append(seps, sep)
    N = np.append(N, n)

min_index = np.argmin(abs(seps - goal))

for i in range(min_index - 1, min_index + 3):
    print(
        f"i = {i}, sep = {seps[i]}, abs(sep - goal) = {abs(seps[i] - goal)}, N = {N[i]}"
    )

print("--------------------------------------------")
for i in range(20, 25):
    print(f"1/{i} = {1/i}")
