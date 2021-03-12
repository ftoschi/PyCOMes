import numpy as np
from numba import jit

@jit(nopython = True, nogil = True, cache = True)
def digitize(p, X):
    
        i = 0
        for x in X:
                if p >= x:
                        i = i+1
                else:
                        return i
        return i
    
@jit(nopython=True, nogil=True, cache=True)
def get_value(E: float, x: np.array, y:np.array): #linear interpolation
    
    if len(x) == 1 and len(y) == 1:
        return y[0]
    
    else:
        index_high = digitize(E, x)
        x0 = x[index_high-1]
        x1 = x[index_high]
        y0 = y[index_high-1]
        y1 = y[index_high]
        res = y0 * (x1 - E)/(x1 - x0) + y1 * (E - x0)/(x1 -x0)
        if index_high == 0:
            res = y[0]
        elif index_high == len(x):
            res = y[-1]
    return res

@jit(nopython=True, nogil=True, cache=True)
def diffuse(dn, E: np.array, diff_t, diff_l, drift, dt:np.array):
    
    normE = np.linalg.norm(E)
    Dt = get_value(normE, diff_t[0], diff_t[1])
    Dl = get_value(normE, diff_l[0], diff_t[1])
    v = get_value(normE, drift[0], drift[1])
    if len(E) == 2:
        norm_dn = np.sqrt(dn[0]**2 + dn[1]**2)
        dt[0] = norm_dn / v # dn is in mm, v in mm/µs -> time in µs

        dn_l = norm_dn #get the longitudinal length of the step, dn_l will be redistributed following Dl
        dn_t = 0
        
        sigma_l = np.sqrt(2 * Dl * dt[0] * 1e-6) * 10 # diffusion constant is in cm²/s and time in µs -> 1e-6 to convert to seconds
        dn_l = dn_l + np.random.normal(0, sigma_l)
        
        sigma_t = np.sqrt(4 * Dt * dt[0] * 1e-6) * 10
        dn_t = dn_t + np.random.normal(0, sigma_t)
        
        E_perp = np.array([1., -E[0]/E[1]], dtype=np.float64)/np.sqrt(1 + (E[0]/E[1])**2)
        dn = - dn_l * E / normE + E_perp * dn_t
        
        return dn
    
    if len(E)==3:
        return dn #to be implemented the 3D way to treat diffusion (longitudinal identical, transversal is 2D problem, 1 uniform phase + 1 gaussian intensity)
