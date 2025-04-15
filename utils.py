from libMobility import PSE, NBody
from scipy import spatial
import numpy as np


def init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_rad, solv_str):

    if solv_str == "PSE":
        solver = PSE("periodic", "periodic", "periodic")
        # split = 1.0
        # split = 4 * n_blobs ** (1 / 3) / Lx
        split = 1 * n_blobs ** (1 / 3) / Lx  # this seemed faster after some tinkering
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
