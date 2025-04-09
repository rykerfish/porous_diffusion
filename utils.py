from libMobility import PSE

def init_solver(n_blobs, Lx, Ly, Lz, kbt, eta, blob_rad):
    solver = PSE("periodic", "periodic", "periodic")
    # split = 4 * n_blobs ** (1 / 3) / Lx
    split = 1 * n_blobs ** (1 / 3) / Lx  # this seemed faster after some tinkering
    solver.setParameters(psi=split, Lx=Lx, Ly=Ly, Lz=Lz, shearStrain=0.0)
    solver.initialize(
        temperature=kbt,
        viscosity=eta,
        hydrodynamicRadius=blob_rad,
        numberParticles=n_blobs,
        needsTorque=False,
    )

    return solver