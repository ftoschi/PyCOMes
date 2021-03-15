import numpy as np
from .Field import *
from .interpolation import *
from .Model import *

class OutOfEdges(Exception):
        pass

class FieldLine():

        def __init__(self, field: Field, edges=None):

                self.field=field
                self.dimension=field.dimension
                self.E_components=np.array(self.field.get_field_components())
                if self.dimension==3 and not all(i in self.field.vars.keys() for i in ['Ex', 'Ey', 'Ez']):
                        raise FieldNotFound
                elif self.dimension==2 and (not all(i in self.field.vars.keys() for i in ['Ex', 'Ey']) and not all(i in self.field.vars.keys() for i in ['Er', 'Ez'])):
                        raise FieldNotFound
                self.X=self.field.X
                self.Y=self.field.Y
                if self.dimension==3:
                        self.Z=self.field.Z

                if edges is None:
                        self.edges=[self.X[0], self.X[-1], self.Y[0], self.Y[-1]]
                        if self.dimension==3:
                                self.edges+=[self.Z[0], self.Z[-1]]
                return

        def set_edges(self, edges):

                edges=list(edges)
                if len(edges)>=self.dimension*2:
                        self.edges=edges[:self.dimension*2]
                elif len(edges)<self.dimension*2:
                        self.edges[:self.dimension*2]=edges
                return edges

        def _is_inside(self):

                conditions=[self.p[0]<self.edges[0], self.p[0]>self.edges[1], self.p[1]<self.edges[2], self.p[1]>self.edges[3]]
                if self.dimension==3:
                        conditions.append(self.p[2]<self.edges[4])
                        conditions.append(self.p[2]>self.edges[5])
                return not any(conditions)

        def set_initial_point(self, p0):

                if self.dimension!=len(p0):
                        raise WrongDimension("the initial point does not match the expected dimension.")
                self.p0=np.array(p0, dtype=np.float64)
                self.p=np.array(p0, dtype=np.float64)
                return
                             
        def closest_point(self):

                coords=[self.X, self.Y]
                if self.dimension==3:
                        coords.append(self.Z)
                return closest_point_grid(self.p, np.array(coords))


        def interpolate(self, params=False):

                if params:
                        self.field.set_parameters(params)

                coords=[self.X, self.Y]
                if self.dimension==3:
                        coords.append(self.Z)

                return interpolate(p, np.array(coords), self.E_components)


        def trajectory(self, dn, diffusion_on=False, print_point=False):

                edges = np.array(self.edges, dtype=np.float64)
                p0 = self.p0.copy()

                try:
                        unit=self.field.vars['z']
                except:
                        unit=self.field.vars['x']
                
                if self.dimension == 3:
                        return trajectory_line_3D(p0, self.X, self.Y, self.Z, self.E_components, dn, edges, print_point=print_point)
                else:
                        return trajectory_line(p0, self.X, self.Y, self.E_components, dn, edges, print_point=print_point)