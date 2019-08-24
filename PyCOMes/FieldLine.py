import matplotlib.pyplot as plt
import numpy as np
from .utils import *
from .Field import *
from .interpolation import *

class WrongDimension(Exception):
	pass

class OutOfEdges(Exception):
	pass

class FieldLine():

	def __init__(self, field: Field, edges=[]):

		self.field=field
		self.dimension=field.dimension
		if self.dimension==3 and not all(i in self.field.vars.keys() for i in ['Ex', 'Ey', 'Ez']):
			raise FieldNotFound
		elif self.dimension==2 and (not all(i in self.field.vars.keys() for i in ['Ex', 'Ey']) and not all(i in self.field.vars.keys() for i in ['Er', 'Ez'])):
			raise FieldNotFound
		self.X=self.field.X
		self.Y=self.field.Y
		if self.dimension==3:
			self.Z=self.field.Z

		if not edges:
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
		self.p0=np.array(p0)
		self.p=np.array(p0)
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

		E_components=self.field.get_field_components()

		return interpolate(p, np.array(coords), np.array(E_components))


	def trajectory(self, dn, diffusion_on=False, print_point=False, plot=False):

		E_components=self.field.get_field_components()

		try:
			unit=self.field.vars['z']
		except:
			unit=self.field.vars['x']
		
		return trajectory_line(self.p, self.X, self.Y, np.array(E_components), dn, self.edges, diffusion_on=diffusion_on, units=unit, print_point=print_point)
