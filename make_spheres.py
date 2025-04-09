import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
from icosphere import icosphere
import time
import utils


def main():
    sphere_radius = 10
    blob_pos, n_spheres, sphere_centers, blob_radius = make_spheres(sphere_radius)
    L = 90  # TODO check value, should probably come from a file
    Lx, Ly, Lz = L, L, L

    print("blob size:", blob_radius)
    print("this should be close to the blob radius:", sphere_radius / 20)

    # create small blobs between porous matrix
    n_colloids = 1000
    colloid_pos = place_colloids(
        sphere_centers, sphere_radius, blob_radius, L, n_colloids
    )

    all_pos = np.append(blob_pos, colloid_pos)
    n_blobs = np.shape(all_pos)[0] // 3
    print("n blobs:", n_blobs)

    kbt = 1.0
    eta = 1.0
    solver = utils.init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_radius)

    solver.setPositions(all_pos)

    forces = np.random.rand(n_blobs * 3) - 0.5
    avg_iter_time(solver, forces)

    # dists = []
    # for i in range(colloid_pos.shape[0] // 3):
    #     test = np.append(sphere_centers.flatten(), colloid_pos[i * 3 : i * 3 + 3])
    #     test_dist = np.min(sp.distance.pdist(test.reshape(-1, 3)))
    #     dists.append(test_dist)
    # print("min dist:", np.min(dists))

    plot_spheres(sphere_centers, sphere_radius, colloid_pos)

    # left to do:
    # 1. write the time stepping code- don't need to use RFDs since it's periodic
    ##### do I use hydrodynamicVelocities for this?
    # 2. put spring forces on blobs in the sphere
    # 3. what data do I save?


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
