import numpy as np
from .utils import *
from .Field import *
from .interpolation import *
from .Model import *
from .FieldLine import *

                    
class Propagate(FieldLine):
    
    def __init__(self, field: Field, diffusion_trans, diffusion_long, drift, edges=[], E_min=0, E_max=200, N=100):
        
        super().__init__(field, edges)
        self.E_min = E_min
        self.E_max = E_max
        self.N = N
        self.diff_t = self._set_model(diffusion_trans)
        self.diff_l = self._set_model(diffusion_long)
        self.drift = self._set_model(drift)
        
    def _set_model(self, prop):
        
        mod_tmp = Model(prop, x_min=self.E_min, x_max=self.E_max, N=self.N)
        return mod_tmp.get_model()
    
    def trajectory(self, dn, print_point=False, plot=False):

                edges = np.array(self.edges, dtype=np.float64)
                p0 = self.p0.copy()
               
                if self.dimension == 3:
                        return trajectory_line_3D(p0, self.X, self.Y, self.Z, self.E_components, dn, self.edges, diff_t=self.diff_t, diff_l=self.diff_l, drift=self.drift, diffuse_on=True, print_point=print_point)
                else:
                        return trajectory_line(p0, self.X, self.Y, self.E_components, dn, edges, diff_t=self.diff_t, diff_l=self.diff_l, drift=self.drift, diffuse_on=True, print_point=print_point)
