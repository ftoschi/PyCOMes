from .diffusion import *


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
def closest_point_grid(p, X, Y):
    length = np.zeros(2, dtype=np.int32)
    length[0] = len(X)
    length[1] = len(Y)

    indeces = np.zeros(2, dtype=np.int32)
    indeces[0] = np.int32(digitize(p[0], X))
    indeces[1] = np.int32(digitize(p[1], Y))

    for i in indeces:
        if i == 0:
            return np.array([-1, -1, -1, -1], dtype=np.int32)
    for i in range(len(indeces)):
        if indeces[i] == length[i]:
            return np.array([-1, -1, -1, -1], dtype=np.int32)

    return_indeces = np.array([indeces[0] - 1, indeces[0], indeces[1] - 1, indeces[1]], dtype=np.int32)
    return return_indeces


@jit(nopython=True, nogil=True, cache=True)
def closest_point_grid_3D(p, X, Y, Z):
    length = np.zeros(3, dtype=np.int32)
    length[0] = len(X)
    length[1] = len(Y)
    length[2] = len(Z)

    indeces = np.zeros(3, dtype=np.int32)
    indeces[0] = np.int32(digitize(p[0], X))
    indeces[1] = np.int32(digitize(p[1], Y))
    indeces[2] = np.int32(digitize(p[2], Z))

    for i in indeces:
        if i == 0:
            return np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
    for i in range(len(indeces)):
        if indeces[i] == length[i]:
            return np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)

    return_indeces = np.array([indeces[0] - 1, indeces[0], indeces[1] - 1, indeces[1], indeces[2] - 1, indeces[2]],
                              dtype=np.int32)
    return return_indeces


@jit(nopython=True, nogil=True, cache=True)
def interpolate_linear(x, x0, x1, f0, f1):
    a = (f1 - f0) / (x1 - x0)
    b = f0 - a * x0
    return a * x + b


@jit(nopython=True, nogil=True, cache=True)
def interpolate_field(p, X, Y, Ex, Ey):
    dimension = 2
    c_p = closest_point_grid(p, X, Y)

    if np.all(c_p == np.array([-1, -1, -1, -1], dtype=np.int32)):
        return np.array([0., 0.], dtype=np.float64)

    xl = c_p[0]
    xu = c_p[1]
    yl = c_p[2]
    yu = c_p[3]

    lenx = len(X)

    LL = np.array([Ex[lenx * yl + xl], Ey[lenx * yl + xl]], dtype=np.float64)
    UL = np.array([Ex[lenx * yu + xl], Ey[lenx * yu + xl]], dtype=np.float64)
    UR = np.array([Ex[lenx * yu + xu], Ey[lenx * yu + xu]], dtype=np.float64)
    LR = np.array([Ex[lenx * yl + xu], Ey[lenx * yl + xu]], dtype=np.float64)

    x1 = X[xl]
    x2 = X[xu]
    y1 = Y[yl]
    y2 = Y[yu]

    C0 = interpolate_linear(p[0], x1, x2, LL, LR)
    C1 = interpolate_linear(p[0], x1, x2, UL, UR)

    C = interpolate_linear(p[1], y1, y2, C0, C1)

    return C


@jit(nopython=True, nogil=True, cache=True)
def interpolate_field_3D(p, X, Y, Z, Ex, Ey, Ez):
    c_p = closest_point_grid_3D(p, X, Y, Z)

    if np.all(c_p == np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)):
        return np.array([0., 0., 0.], dtype=np.float64)

    xl = c_p[0]
    xu = c_p[1]
    yl = c_p[2]
    yu = c_p[3]
    zl = c_p[4]
    zu = c_p[5]

    lenx = len(X)
    leny = len(Y)
    lenz = len(Z)

    i = xl + lenx * yl + lenx * leny * zl
    LLL = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xl + lenx * yl + lenx * leny * zu
    LLU = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xl + lenx * yu + lenx * leny * zl
    LUL = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xl + lenx * yu + lenx * leny * zu
    LUU = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xu + lenx * yl + lenx * leny * zl
    ULL = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xu + lenx * yl + lenx * leny * zu
    ULU = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xu + lenx * yu + lenx * leny * zl
    UUL = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)
    i = xu + lenx * yu + lenx * leny * zu
    UUU = np.array([Ex[i], Ey[i], Ez[i]], dtype=np.float64)

    x1 = X[xl]
    x2 = X[xu]
    y1 = Y[yl]
    y2 = Y[yu]
    z1 = Z[zl]
    z2 = Z[zu]

    C00 = interpolate_linear(p[0], x1, x2, LLL, ULL)
    C01 = interpolate_linear(p[0], x1, x2, LLU, ULU)
    C10 = interpolate_linear(p[0], x1, x2, LUL, UUL)
    C11 = interpolate_linear(p[0], x1, x2, LUU, UUU)

    C0 = interpolate_linear(p[1], y1, y2, C00, C01)
    C1 = interpolate_linear(p[1], y1, y2, C10, C11)

    C = interpolate_linear(p[2], z1, z2, C0, C1)

    return C


@jit(nopython=True, nogil=True, cache=True)
def is_inside(p, edges):
    conditions = np.array([p[0] < edges[0], p[0] > edges[1], p[1] < edges[2], p[1] > edges[3]], dtype=np.bool8)
    if len(p) == 3:
        conditions = np.append(conditions, np.asarray([p[2] < edges[4], p[2] > edges[5]], dtype=np.bool8))
    return not conditions.any()


@jit(nopython=True, nogil=True, cache=True)
def trajectory_line(p, X, Y, Ex, Ey, dn, edges, diff_t=None, diff_l=None, drift=None,
                    diffuse_on=False, axisymmetry=True, print_point=False):

    length = 0.
    time = 0.
    x_tmp = np.array([p[0]], dtype=np.float64)
    y_tmp = np.array([p[1]], dtype=np.float64)
    t_tmp = np.array([0.], dtype=np.float64)
    if axisymmetry:
        theta_tmp = np.random.uniform(-np.pi, np.pi)
    if (diff_t is None or diff_l is None or drift is None) and diffuse_on:
        raise ValueError(': no diffusion or drift speed specified, diffusion not possible.')

    while is_inside(p, edges):
        E = interpolate_field(p, X, Y, Ex, Ey)

        # checking if the field value makes sense
        if np.isnan(E[0]):
            break
        normE = np.float64(np.linalg.norm(E))
        if normE == 0.:
            break

        # move along E with step size dn
        dp = -dn * E / normE
        dt = np.array([0.], dtype=np.float64)
        if diffuse_on:
            if not axisymmetry:
                p = diffuse(p, dp, E, diff_t, diff_l, drift, dt)
                length = length + np.sqrt(dp[0] ** 2 + dp[1] ** 2)
                time += dt[0]
            else:
                # as this funciton should handle only 2D, not checking dimensionality
                p = np.array([p[0] * np.cos(theta_tmp), p[0] * np.sin(theta_tmp), p[1]],
                             dtype=np.float64)
                p = diffuse(p, dp, E, diff_t, diff_l, drift, dt)
                theta_tmp = np.arctan2(p[1], p[0])
                p = np.array([np.sqrt(p[0]**2 + p[1]**2), p[2]], dtype=np.float64)
        else:
            length += dn
            p = p + dp
        x_tmp = np.concatenate((x_tmp, np.array([p[0]], dtype=np.float64)))
        y_tmp = np.concatenate((y_tmp, np.array([p[1]], dtype=np.float64)))
        t_tmp = np.concatenate((t_tmp, np.array([time], dtype=np.float64)))
        if print_point:
            print(p, time)

    return x_tmp, y_tmp, t_tmp


@jit(nopython=True, nogil=True, cache=True)
def trajectory_line_3D(p, X, Y, Z, Ex, Ey, Ez, dn, edges, diffusion_on=False, print_point=False):
    dn = float(dn)
    length = 0.
    time = 0.

    x_tmp = np.array([p[0]])
    y_tmp = np.array([p[1]])
    z_tmp = np.array([p[2]])

    while is_inside(p, edges):
        E = interpolate_field_3D(p, X, Y, Z, Ex, Ey, Ez)
        if np.isnan(E[0]):
            break
        normE = np.linalg.norm(E)
        dp = -dn * E / normE
        if diffusion_on:
            pass  # dp = diffuse(dp, E, units)
            # length += np.linalg.norm(dp)
        else:
            length += dn
        p[0] += dp[0]
        p[1] += dp[1]
        p[2] += dp[2]

        x_tmp = np.concatenate((x_tmp, np.array([p[0]])))
        y_tmp = np.concatenate((y_tmp, np.array([p[1]])))
        z_tmp = np.concatenate((z_tmp, np.array([p[2]])))
        if print_point:
            print(p)

    return x_tmp, y_tmp, z_tmp
