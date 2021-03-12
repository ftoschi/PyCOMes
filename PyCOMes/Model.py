import numpy as np

class WrongDimension(Exception):
        pass

class Model():
    
    def __init__(self, prop, x_min=0, x_max=200, N=100):
        
        self._set_model(prop, x_min=x_min, x_max=x_max, N=N)
        
    def _set_model(self, prop, x_min=0, x_max=200, N=100):

        if isinstance(prop, float) or isinstance(prop, int):
            self.x = np.array([1.], dtype=np.float64)
            self.y = np.array([prop], dtype=np.float64)

        elif hasattr(prop, '__len__'):
            prop = np.array(prop, dtype=np.float64)
            if prop.shape[0] == 2:
                self.x = np.array(prop[0])
                self.y = np.array(prop[1])
            elif prop.shape[1] == 2:
                self.x = np.array(prop[:,0])
                self.y = np.array(prop[:,1])
            else:
                raise WrongDimension("the inserted model array has wrong dimensions, it must a (2,N)/(N,2) array or list")

        elif hasattr(prop, '__call__'):
            x = np.linspace(x_min, x_max, N)
            self.x = x
            self.y = prop(x)

        else:
            raise WrongDimension("the inserted model has wrong dimensions, it must be a scalar, a function or a (2,N)/(N,2) array or list")
            
    def get_model(self):
        
        return np.array([self.x, self.y])