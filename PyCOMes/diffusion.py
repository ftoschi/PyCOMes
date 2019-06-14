import numpy as np
import numba

@jit(nopython=True, nogil=True, cache=True)
def get_diffusion(E):
    return #insert approximate function for diffusion (two components, long and trans)

@jit(nopython=True, nogil=True, cache=True)
def get_drift(E):
    return #drift speed

@jit(nopython=True, nogil=True, cache=True)
def diffuse(dn: np.array, E, units):
    Dt,Dl=get_diffusion(E)
    v=get_drift(E) #fix the units
    norm_dn=np.linalg.norm(dn)
    dt=norm_dn/v #check for the units

    dn_l=np.dot(dn,E)/np.linalg.norm(E) #get the longitudinal length of the step, dn_l will be redistributed following Dl
    sigma_E=np.sqrt(2*Dl*dt)
    dn_l=dn_l+np.random.normal(0,sigma_E)

    if len(E)==2:
        dn_t=np.sqrt(norm_dn**2-dn_l**2)
        #shouldn't it be null? Because dn is built following E. In case it is easier <3
        sigma_r=np.sqrt(2*Dt*dt)
        dn_t=dn_t+np.random.normal(0,sigma_r)
        E_perp=np.array(1,-E[0]/E[1])/np.sqrt(1+(E[0]/E[1])**2)
        return dn_l*E/np.linalg.norm(E)+E_perp*dn_t

    if len(E)==3:
        return dn #to be implemented the 3D way to treat diffusion (longitudinal identical, transversal is 2D problem, 1 uniform phase + 1 gaussian intensity)
