# heat.py
import numpy as np

def solve_heat(cfg):
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    dt, steps = cfg["dt"], cfg["steps"]
    alpha = 1.0  # thermal diffusivity

    # Initial condition: hot plane at x = 0
    u = np.zeros((nx, ny, nz))
    u[0, :, :] = 1.0

    for _ in range(steps):
        u_old = u.copy()

        # 3D discrete Laplacian
        lap = (
            u_old[:-2, 1:-1, 1:-1] +
            u_old[2:, 1:-1, 1:-1] +
            u_old[1:-1, :-2, 1:-1] +
            u_old[1:-1, 2:, 1:-1] +
            u_old[1:-1, 1:-1, :-2] +
            u_old[1:-1, 1:-1, 2:] -
            6 * u_old[1:-1, 1:-1, 1:-1]
        )

        # Heat diffusion update
        u[1:-1, 1:-1, 1:-1] += alpha * dt * lap

    return u
