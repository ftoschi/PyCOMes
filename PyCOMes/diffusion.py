import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def digitize(p, X):
    i = 0
    for x in X:
        if p >= x:
            i = i + 1
        else:
            return i
    return i


@jit(nopython=True, nogil=True, cache=True)
def get_value(E: float, x: np.array, y: np.array):  # linear interpolation

    if len(x) == 1 and len(y) == 1:
        return y[0]

    else:
        index_high = digitize(E, x)
        x0 = x[index_high - 1]
        x1 = x[index_high]
        y0 = y[index_high - 1]
        y1 = y[index_high]
        res = y0 * (x1 - E) / (x1 - x0) + y1 * (E - x0) / (x1 - x0)
        if index_high == 0:
            res = y[0]
        elif index_high == len(x):
            res = y[-1]
    return res


@jit(nopython=True, nogil=True, cache=True)
def diffuse(p, dn, E, diff_t, diff_l, drift, dt):
    if diff_l is None:
        diff_l = np.array([[0], [0]], dtype=np.float64)
    if diff_t is None:
        diff_t = np.array([[0], [0]], dtype=np.float64)
    if drift is None:
        drift = np.array([[0], [1]], dtype=np.float64)

    normE = np.linalg.norm(E)
    Dt = get_value(normE, diff_t[0], diff_t[1])
    Dl = get_value(normE, diff_l[0], diff_t[1])
    v = get_value(normE, drift[0], drift[1])

    norm_dn = np.linalg.norm(dn)
    dt[0] = norm_dn / v  # dn is in mm, v in mm/µs -> time in µs

    dn_l = norm_dn  # get the longitudinal length of the step, dn_l will be redistributed following Dl
    dn_t_x = 0.  # as we followed the field line, the transverse component must be null
    dn_t_y = 0.

    # diffusion constant is in cm²/s and time in µs -> 1e-6 to convert to seconds
    sigma_l = np.sqrt(2 * Dl * dt[0] * 1e-6) * 10
    dn_l = dn_l + np.random.normal(0, sigma_l)

    sigma_t = np.sqrt(2 * Dt * dt[0] * 1e-6) * 10
    dn_t_x = dn_t_x + np.random.normal(0, sigma_t)
    dn_t_y = dn_t_y + np.random.normal(0, sigma_t)

    if len(E) == 2:
        E_perp = np.array([1., -E[0] / E[1]], dtype=np.float64) / np.sqrt(1 + (E[0] / E[1]) ** 2)
        p = p - dn_l * E / normE + E_perp * dn_t_x
    else:
        E_perp_x = np.array([0, -E[2] / E[1], 1], dtype=np.float64) / np.sqrt(1 + (E[2] / E[1]) ** 2)
        E_perp_y = np.array([0, 1, -E[1] / E[2]], dtype=np.float64) / np.sqrt(1 + (E[1] / E[2]) ** 2)
        p = p - dn_l * E / normE + E_perp_x * dn_t_x + E_perp_y * dn_t_y

    return p
