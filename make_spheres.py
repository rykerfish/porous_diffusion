import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from icosphere import icosphere
import time
import utils
from numba import njit, prange
from libMobility import PSE


def main():
    sphere_radius = 100e-6  # m
    rigid_blob_x0, n_spheres, sphere_centers, blob_radius = make_spheres(sphere_radius)
    L = 4.32 * 2 * sphere_radius  # TODO check value, should probably come from a file
    Lx, Ly, Lz = L, L, L
    L_arr = np.array([Lx, Ly, Lz])
    n_rigid_blobs = rigid_blob_x0.shape[0] // 3

    print("n_rigid_blobs:", n_rigid_blobs)
    print("blob size:", blob_radius)
    print("this should be close to the blob radius:", sphere_radius / 20)

    # create small blobs between porous matrix
    n_colloids = 1000
    colloid_pos = place_colloids(
        sphere_centers, sphere_radius, blob_radius, L, n_colloids
    )

    all_pos = np.append(rigid_blob_x0, colloid_pos)
    n_blobs = np.shape(all_pos)[0] // 3
    print("n blobs:", n_blobs)

    kbt = 4.11e-21  # J
    eta = 1.0e-3  # Pa.s
    solver = utils.init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_radius, "PSE")
    precision = np.float32 if PSE.precision == "float" else np.float64

    k_spring = 1e6 * kbt
    U0 = 4 * kbt
    debye = 0.1 * blob_radius

    # forces = np.random.rand(n_blobs * 3) - 0.5
    # avg_iter_time(solver, forces)

    # dists = []
    # for i in range(colloid_pos.shape[0] // 3):
    #     test = np.append(sphere_centers.flatten(), colloid_pos[i * 3 : i * 3 + 3])
    #     test_dist = np.min(sp.distance.pdist(test.reshape(-1, 3)))
    #     dists.append(test_dist)
    # print("min dist:", np.min(dists))

    # plot_spheres(sphere_centers, sphere_radius, colloid_pos)

    # T_final = 4 * 3600  # in seconds, 4 hours
    dt = 0.001  # in seconds TODO fix
    T_final = 100 * dt  # TODO temp
    n_steps = int(T_final / dt)
    print("n steps:", n_steps)

    steric_time = 0
    for i in range(n_steps):
        print("step:", i)
        solver.setPositions(all_pos)

        forces = np.zeros((n_blobs * 3), dtype=precision)
        forces[0 : 3 * n_rigid_blobs] += -k_spring * (
            all_pos[0 : 3 * n_rigid_blobs] - rigid_blob_x0
        )

        start = time.time()
        forces += blob_blob_force_numba(
            L_arr, blob_radius, all_pos, U0, debye, n_rigid_blobs, precision
        ).flatten()
        end = time.time()
        steric_time += end - start

        # (I think I can use an Euler-Maryama and hydrodynamicVelocities, check PSE paper)
        v, _ = solver.hydrodynamicVelocities(forces)
        all_pos += dt * v.flatten()

        print(
            np.max(rigid_blob_x0 - all_pos[0 : 3 * n_rigid_blobs]),
            np.min(rigid_blob_x0 - all_pos[0 : 3 * n_rigid_blobs]),
        )
        # print(np.max(all_pos), np.min(all_pos))

        # if i % 100 == 0:
        #     plot_spheres(sphere_centers, sphere_radius, all_pos)

    print("average steric time:", steric_time / n_steps)
    print("total time:", steric_time)


########## version using symmetry and ignoring rigid-rigid interactions
@njit(parallel=True, fastmath=True)
def blob_blob_force_numba(
    L, a, r_vectors, repulsion_strength, debye_length, n_exclude, precision
):
    """
    This function compute the force between two blobs
    with vector between blob centers r.

    In this example the force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  if z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius
    n_exclude = number of blobs to skip sterics on.
      This is useful when using rigid multiblobs to not include steric interactions between
      those particles, but requires them to be at the beginning of the r_vectors array.
    """

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    force = np.zeros((N, 3)).astype(precision)

    for i in prange(N):
        # this logic skips rigid-rigid interactions and self-interactions (i=j)
        # the goal is to only include rigid-colloid and colloid-colloid interactions
        if i < n_exclude:
            j_start = n_exclude
        else:  # start above i since (i,j) = -(j,i) which is included below
            j_start = i + 1

        for j in range(j_start, N):

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                if L[k] > 0:
                    dr[k] -= (
                        int(dr[k] / L[k] + 0.5 * (int(dr[k] > 0) - int(dr[k] < 0)))
                        * L[k]
                    )

            # Compute force
            r_norm = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])
            #   r_hat = dr/r_norm

            offset = 2.0 * a
            if r_norm > offset:
                coeff = (
                    -(repulsion_strength / debye_length)
                    * np.exp(-(r_norm - offset) / debye_length)
                    / np.maximum(r_norm, 1.0e-16)
                )
            else:
                coeff = -(repulsion_strength / debye_length) / np.maximum(
                    r_norm, 1.0e-16
                )

            iter_force = coeff * dr

            force[i] += iter_force
            force[j] -= iter_force

    return force


def plot_spheres(sphere_centers, sphere_radius, colloid_pos=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    for i in range(len(sphere_centers)):
        x = sphere_centers[i][0]
        y = sphere_centers[i][1]
        z = sphere_centers[i][2]
        # Generate a sphere using parametric equations
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + x
        y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + y
        z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(x, y, z, color="b", alpha=0.3)

    if colloid_pos is not None:
        ax.scatter(
            colloid_pos[0::3],
            colloid_pos[1::3],
            colloid_pos[2::3],
            color="r",
            alpha=0.5,
        )

    plt.show()


def avg_iter_time(solver, forces):
    for i in range(5):
        test, _ = solver.Mdot(forces)
    test_times = []
    for i in range(10):
        start = time.time()
        test, _ = solver.Mdot(forces)
        end = time.time()
        test_times.append(end - start)
    test_times = np.array(test_times)
    print("average time:", np.mean(test_times))


def place_colloids(sphere_centers, sphere_radius, blob_radius, L, n_colloids):
    kd_box = [L, L, L]  # basically not periodic in the last dim
    print(np.max(sphere_centers, axis=0))
    print(np.min(sphere_centers, axis=0))
    kd = sp.KDTree(
        data=sphere_centers,
        copy_data=True,
        boxsize=kd_box,
    )

    colloid_pos = np.zeros((n_colloids, 3))
    n_placed = 0
    while n_placed < n_colloids:
        rand_pos = (np.random.rand(3)) * L
        pts = kd.query_ball_point(
            rand_pos,
            sphere_radius + 2.5 * blob_radius,
            p=2,
        )
        if len(pts) > 0:
            continue

        colloid_pos[n_placed] = rand_pos
        n_placed += 1

    return colloid_pos.flatten()


def make_spheres(sphere_radius):
    sphere_diam = 2 * sphere_radius

    dat = np.fromfile("dat/small.xyzd")
    n_spheres = len(dat) // 4

    sphere_centers = np.array([dat[i] for i in range(len(dat)) if i % 4 != 3])
    # diameters = sphere_diam * dat[3::4]

    sphere_centers *= sphere_diam
    sphere_centers = sphere_centers.reshape((n_spheres, 3))

    temp_ico, _ = icosphere(18)
    ico_pos = temp_ico * sphere_radius
    blob_sep = np.min(sp.distance.pdist(ico_pos))

    pos = np.array([])
    for i in range(0, n_spheres):
        shift = sphere_centers[i, :]

        ico_pos, _ = icosphere(18)
        ico_pos *= sphere_radius  # could change to use diameter array for polydisperse?
        ico_pos += shift
        pos = np.append(pos, ico_pos.flatten())

    return pos, n_spheres, sphere_centers, blob_sep


if __name__ == "__main__":
    main()
