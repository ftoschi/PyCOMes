from .FieldLine import *

diffusion_t_LXe = np.array([[19.8327, 37.68865, 74.36972, 185.76201, 372.43775, 609.02498],
                            [48.43326, 56.51715, 54.61218, 56.34393, 55.64231, 60.76323]])
diffusion_l_LXe = np.array([[15.86777, 30.41322, 54.87603, 78.93119, 103.01032, 151.16859,
                             200.78619, 297.83240, 395.60827, 492.65448],
                            [82.24138, 63.27586, 46.37931, 37.80507, 31.33554, 26.22515,
                             24.32234, 22.67777, 22.29721, 19.70728]])
drift_speed_LXe = np.array([[11.02362, 13.12336, 14.69816, 19.69292, 37.60378, 74.61954,
                             186.86087, 380.29806, 566.57092, 615.52724],
                            [0.31564, 0.37337, 0.41993, 0.53285, 0.90554, 1.24451, 1.51567,
                             1.70609, 1.85028, 1.87579]])


class Propagate(FieldLine):

    def __init__(self, field: Field, diffusion_trans=diffusion_t_LXe,
                 diffusion_long=diffusion_l_LXe, drift=drift_speed_LXe,
                 edges=None, E_min=0, E_max=200, N=100, axisymmetry=True):

        super().__init__(field, edges, axisymmetry=axisymmetry)
        self.diffusion_on = True
        self.E_min = E_min
        self.E_max = E_max
        self.N = N
        self.diff_t = self._set_model(diffusion_trans)
        self.diff_l = self._set_model(diffusion_long)
        self.drift = self._set_model(drift)

    def _set_model(self, prop):

        mod_tmp = Model(prop, x_min=self.E_min, x_max=self.E_max, N=self.N)
        return mod_tmp.get_model()
