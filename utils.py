from libMobility import PSE, NBody
from scipy import spatial
import numpy as np
from numba import njit, prange
import json


def init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_rad, solv_str):

    if solv_str == "PSE":
        solver = PSE("periodic", "periodic", "periodic")
        # split = 1.0
        # split = 4 * n_blobs ** (1 / 3) / Lx
        split = 1.5 * n_blobs ** (1 / 3) / Lx  # this seemed faster after some tinkering
        solver.setParameters(psi=split, Lx=Lx, Ly=Ly, Lz=Lz, shearStrain=0.0)

    elif solv_str == "NBody":
        solver = NBody("open", "open", "open")
        solver.setParameters()

    solver.initialize(
        temperature=kbt,
        viscosity=eta,
        hydrodynamicRadius=blob_rad,
        numberParticles=n_blobs,
        needsTorque=False,
    )

    return solver


def build_neighbor_list(r_vectors, L, r_cut, eps=0.0):
    # rvecs2D = r_vectors[:, 0:dim]
    # L2D = L[0:dim]
    # note: could be generalized to other dimensions by including a dim parameter and slicing as above

    r_vectors = periodize_r_vecs(r_vectors, L, np.shape(r_vectors)[0])

    # r_vectors = np.reshape(r_vectors, (-1, 3))

    r_tree = (
        spatial.cKDTree(  # TODO benchmark the balanced_tree and compact_node options
            r_vectors, boxsize=L, balanced_tree=False, compact_nodes=False
        )
    )

    pairs = r_tree.query_ball_point(
        r_vectors, r_cut, return_sorted=False, workers=1, eps=eps
    )  # eps has a large effect on performance and can affect accuracy if set incorrectly

    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter(
        (item for sublist in pairs for item in sublist), dtype=int
    )
    return offsets, list_of_neighbors


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    # r_vecs = np.reshape(r_vecs, (Nb, 3))
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs


# only ever append to filename but as csv of flattened xyz
# note that if the file already exists it'll just start adding on to the end
def save_pos(pos, current_time, filename):
    with open(filename, "a") as f:
        row = [current_time] + pos.flatten().tolist()
        f.write(",".join(map(str, row)) + "\n")


def save_params_json(params):
    with open("params.json", "w") as f:
        json.dump(params, f, indent=4)
    print("Saved parameters to params.json")


def log(message, filename="log.txt"):
    with open(filename, "a") as f:
        f.write(message + "\n")
