import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from icosphere import icosphere
import time
import utils
from numba import njit, prange
from libMobility import PSE

fname = "colloid_pos.csv"

def main():
    sphere_radius = 100  # um
    in_file = "dat/small.xyzd"
    rigid_blob_x0, n_spheres, sphere_centers, blob_radius = make_spheres(
        sphere_radius, in_file
    )
    L = (
        4.32 * 2 * sphere_radius
    )  # TODO should probably come from the sphere positions file
    Lx, Ly, Lz = L, L, L
    L_arr = np.array([Lx, Ly, Lz])
    n_rigid_blobs = rigid_blob_x0.shape[0] // 3
    rigid_blob_x0 = np.reshape(rigid_blob_x0, (n_rigid_blobs, 3))

    print("L:", L)
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
    all_pos = np.reshape(all_pos, (n_blobs, 3))
    x0 = all_pos.copy()
    print("n blobs:", n_blobs)

    kbt = 0.004  # aJ
    eta = 1.0e-3  # Pa.s
    solver = utils.init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_radius, "PSE")
    precision = np.float32 if PSE.precision == "float" else np.float64

    k_spring = 1e2 * kbt / (blob_radius**2)
    U0 = 4 * kbt
    debye = 0.1 * blob_radius
    n_cutoff = 5  # number of debye lengths to include in the cutoff
    r_cut = (
        2 * blob_radius + n_cutoff * debye
    )  # TODO check value of sterics at cutoff distance

    eps = 0.0  # changing eps from 0.0 might help performance but miss some interactions in neighbor list
    delta = 0.25  # in units of blob radius
    offsets, neighbor_list = utils.build_neighbor_list(
        all_pos, L_arr, r_cut + delta * blob_radius, eps
    )

    # spring timescale
    T_k = 6 * np.pi * eta * blob_radius / k_spring
    print("spring timescale:", T_k)
    print("avg displacement:", np.sqrt(kbt / k_spring))
    # print("D:", kbt / (6 * np.pi * eta * blob_radius))

    dt = 1.0e-3 * T_k
    print("dt:", dt)

    T_final = 3600  # in seconds, 1 hour
    n_steps = int(T_final / dt)
    print("n steps:", n_steps)
    n_save = np.floor(0.5 / dt)  # save every half second
    print("n save:", n_save)

    params = {
        "n_rigid_blobs": n_rigid_blobs,
        "n_blobs": n_blobs,
        "blob_radius": blob_radius,
        "k_spring": k_spring,
        "U0": U0,
        "debye": debye,
        "r_cut": r_cut,
        "delta": delta,
        "eps": eps,
        "n_cutoff": n_cutoff,
        "dt": dt,
        "T_final": T_final,
        "L": Lx,
        "kbt": kbt,
        "eta": eta,
        "sphere_radius": sphere_radius,
        "n_colloids": n_colloids,
    }

    utils.save_params_json(params)

    n_out = 500
    for i in range(n_steps):
        if i % n_out == 1:
            with open("log.txt", "a") as f:
                utils.log(f"step: {i}\n")
                rigid_drift = np.max(
                    np.linalg.vector_norm(
                        all_pos[0:n_rigid_blobs, :] - rigid_blob_x0, axis=1
                    )
                )
                utils.log
                f.write(f"max rigid drift: {rigid_drift}\n")

        if i % n_save == 0:
            print("saving colloid pos at step:", i)
            colloid_pos = all_pos[n_rigid_blobs:, :].flatten()
            utils.save_pos(colloid_pos, i * dt, fname)

        forces = np.zeros((n_blobs, 3), dtype=precision)
        spring_force = -k_spring * (all_pos[0:n_rigid_blobs, :] - rigid_blob_x0)
        forces[0:n_rigid_blobs, :] += spring_force

        sterics = blob_blob_sterics(
            all_pos,
            L_arr,
            blob_radius,
            U0,
            debye,
            neighbor_list,
            offsets,
            n_rigid_blobs,
        )
        forces += sterics

        solver.setPositions(all_pos)
        v, _ = solver.hydrodynamicVelocities(forces)
        all_pos += dt * v

        if np.max(dt * v) > 0.25 * blob_radius:
            print("dt maybe too large, decrease dt")
            # break

        max_delta_pos = np.max(np.linalg.vector_norm(all_pos - x0, axis=1))
        if max_delta_pos > delta * blob_radius:
            print("rebuilding neighbor list")
            offsets, neighbor_list = utils.build_neighbor_list(
                all_pos, L_arr, r_cut + delta * blob_radius, eps
            )
            x0 = all_pos.copy()


@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    L,
    a,
    repulsion_strength,
    debye_length,
    list_of_neighbors,
    offsets,
    n_exclude,
):
    """
    The force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius
    """

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    force = np.zeros((N, 3))

    for i in prange(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = list_of_neighbors[offsets[i] + kk]
            if i == j:
                continue
            if i < n_exclude and j < n_exclude:  # skip rigid-rigid interactions
                continue

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

            for k in range(3):
                offset = 2.0 * a
                if r_norm > (offset):
                    force[i, k] += (
                        -(
                            (repulsion_strength / debye_length)
                            * np.exp(-(r_norm - (offset)) / debye_length)
                            / np.maximum(r_norm, 1.0e-12)
                        )
                        * dr[k]
                    )
                else:
                    force[i, k] += (
                        -(
                            (repulsion_strength / debye_length)
                            / np.maximum(r_norm, 1.0e-12)
                        )
                        * dr[k]
                    )

    return force


def plot_spheres(sphere_centers, sphere_radius, colloid_pos=None, fname=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    for i in range(len(sphere_centers)):
        x = sphere_centers[i][0]
        y = sphere_centers[i][1]
        z = sphere_centers[i][2]
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

    plt.savefig(fname)
    plt.close(fig)


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
    kd_box = [L, L, L]
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
            sphere_radius + 3 * blob_radius,
            p=2,
        )
        if len(pts) > 0:
            continue

        colloid_pos[n_placed] = rand_pos
        n_placed += 1

    return colloid_pos.flatten()


def make_spheres(sphere_radius, infile):
    sphere_diam = 2 * sphere_radius

    dat = np.fromfile(infile)
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
