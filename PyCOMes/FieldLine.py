from .Field import *
from .interpolation import *
from .Model import *


class OutOfEdges(Exception):
    pass


class FieldLine:

    def __init__(self, field: Field, edges=None, axisymmetry=True):

        self.field = field
        self.dimension = field.dimension
        self.E_components = np.array(self.field.get_field_components())
        self.axisymmetry = axisymmetry
        self.diffusion_on = False
        if self.dimension == 3 and not all(i in self.field.vars.keys() for i in ['Ex', 'Ey', 'Ez']):
            raise FieldNotFound
        elif self.dimension == 2 and (not all(i in self.field.vars.keys() for i in ['Ex', 'Ey']) and not all(
                i in self.field.vars.keys() for i in ['Er', 'Ez'])):
            raise FieldNotFound
        self.X = np.asarray(self.field.X)
        self.Y = np.asarray(self.field.Y)
        if self.dimension == 3:
            self.Z = np.asarray(self.field.Z)

        self.diff_t = None
        self.diff_l = None
        self.drift = None

        if edges is None:
            self.edges = np.array([self.X[0], self.X[-1], self.Y[0], self.Y[-1]])
            if self.dimension == 3:
                self.edges = np.append(self.edges, [self.Z[0], self.Z[-1]])
            self.edges = np.asarray(self.edges)
        else:
            self.edges = edges
        self.radial_edges = False
        return
    
    def __call__(self, dn, theta=0., print_point=False, step_limit=None):
        
        return self.trajectory(dn, theta=0., print_point=print_point, step_limit=step_limit)

    def set_edges(self, edges):

        edges = list(edges)
        if len(edges) >= self.dimension * 2:
            self.edges = edges[:self.dimension * 2]
        elif len(edges) < self.dimension * 2:
            self.edges[:self.dimension * 2] = edges
        self.radial_edges = False
            
    def set_rz_edges(self, edges):
      
        assert (self.dimension == 3) and (len(edges) == 4)
        self.edges = edges
        self.radial_edges = True

    def _is_inside(self):

        if(self.radial_edges):
            conditions = [self.p[0]**2 + self.p[1]**2 < self.edges[0]**2,
                          self.p[0]**2 + self.p[1]**2 > self.edges[1]**2,
                          self.p[2] < self.edges[2], self.p[2] > self.edges[3]]
        else:
            conditions = [self.p[0] < self.edges[0], self.p[0] > self.edges[1], self.p[1] < self.edges[2],
                          self.p[1] > self.edges[3]]
            if self.dimension == 3:
                conditions.append(self.p[2] < self.edges[4])
                conditions.append(self.p[2] > self.edges[5])
        return not any(conditions)

    def set_initial_point(self, p0):

        if self.dimension != len(p0):
            raise WrongDimension(": the initial point does not match the expected dimension.")
        self.p0 = np.array(p0, dtype=np.float64)
        self.p = np.array(p0, dtype=np.float64)
        return

    def trajectory(self, dn, theta=0., print_point=False, step_limit=None):

        p0 = self.p0.copy()
        dn = np.float64(dn)
        theta = np.float64(theta)

        if self.dimension == 3:
            traj = trajectory_line_3D(np.asarray(p0, dtype=np.float64),
                                      np.asarray(self.X, dtype=np.float64),
                                      np.asarray(self.Y, dtype=np.float64),
                                      np.asarray(self.Z, dtype=np.float64),
                                      np.asarray(self.E_components[0], dtype=np.float64),
                                      np.asarray(self.E_components[1], dtype=np.float64),
                                      np.asarray(self.E_components[2], dtype=np.float64),
                                      dn,
                                      np.asarray(self.edges, dtype=np.float64),
                                      diff_t=self.diff_t,
                                      diff_l=self.diff_l,
                                      drift=self.drift,
                                      diffuse_on=self.diffusion_on,
                                      print_point=print_point,
                                      step_limit=step_limit)

            return traj
        else:
            traj = trajectory_line(np.asarray(p0, dtype=np.float64),
                                   np.asarray(self.X, dtype=np.float64),
                                   np.asarray(self.Y, dtype=np.float64),
                                   np.asarray(self.E_components[0], dtype=np.float64),
                                   np.asarray(self.E_components[1], dtype=np.float64),
                                   dn,
                                   np.asarray(self.edges, dtype=np.float64),
                                   axisymmetry=self.axisymmetry,
                                   diff_t=self.diff_t,
                                   diff_l=self.diff_l,
                                   drift=self.drift,
                                   diffuse_on=self.diffusion_on,
                                   theta=theta,
                                   print_point=print_point,
                                   step_limit=step_limit)
            if self.axisymmetry:
                return traj
            else:
                return traj[0], traj[2], traj[3]
