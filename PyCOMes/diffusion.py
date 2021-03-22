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
def diffuse(p, dp, E, diff_t, diff_l, drift, dt):
    
    if diff_l is None:
        diff_l = np.array([[0], [0]], dtype=np.float64)
    if diff_t is None:
        diff_t = np.array([[0], [0]], dtype=np.float64)
    if drift is None:
        drift = np.array([[0], [1]], dtype=np.float64)

    normE = np.linalg.norm(E)
    Dt = get_value(normE, diff_t[0], diff_t[1])
    Dl = get_value(normE, diff_l[0], diff_l[1])
    v = get_value(normE, drift[0], drift[1])

    norm_dp = np.linalg.norm(dp)
    dt[0] = norm_dp / v  # dp is in mm, v in mm/µs -> time in µs

    dp_l = norm_dp  # get the longitudinal length of the step, dp_l will be redistributed following Dl
    dp_t_x = 0.  # as we followed the field line, the transverse component must be null
    dp_t_y = 0.

    # diffusion constant is in cm²/s and time in µs -> 1e-6 to convert to seconds
    sigma_l = np.sqrt(2 * Dl * dt[0] * 1e-6) * 10
    dp_l = dp_l + np.random.normal(0, sigma_l)

    sigma_t = np.sqrt(2 * Dt * dt[0] * 1e-6) * 10
    dp_t_x = dp_t_x + np.random.normal(0, sigma_t)
    dp_t_y = dp_t_y + np.random.normal(0, sigma_t)

    if len(p) == 2:
        E_perp = np.array([-E[1], E[0]], dtype=np.float64) / normE
        p = p - dp_l * E / normE + E_perp * dp_t_x
    else:
        v_rndm = np.asarray(np.random.uniform(0, 1, 3), dtype=np.float64)
        E_perp_x = np.cross(v_rndm, E)
        E_perp_x = E_perp_x / np.linalg.norm(E_perp_x)
        E_perp_y = np.cross(E, E_perp_x) / np.linalg.norm(E)
        p = p - dp_l * E / normE + E_perp_x * dp_t_x + E_perp_y * dp_t_y

    return p
