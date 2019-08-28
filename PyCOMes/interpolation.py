from numba import jit
import numpy as np
from numba.targets.rangeobj import range_iter_len
import sys

@jit(nopython=True, nogil=True, cache=True)
def digitize(p, X):
    
	i=0
	for x in X:
		if p>=x:
			i=i+1
		else:
			return i
	return i
        
@jit(nopython=True, nogil=True, cache=True)
def closest_point_grid(p, X, Y):

	length=np.zeros(2, dtype=np.int32)
	length[0]=len(X)
	length[1]=len(Y)

	indeces=np.zeros(2, dtype=np.int32)
	indeces[0]=np.int32(digitize(p[0], X))
	indeces[1]=np.int32(digitize(p[1], Y))

	for i in indeces:
		if i==0:
			return np.array([-1,-1,-1,-1], dtype=np.int32)
	for i in range(len(indeces)):
		if indeces[i]==length[i]:
			return np.array([-1,-1,-1,-1], dtype=np.int32)

	return_indeces=np.array([indeces[0]-1, indeces[0], indeces[1]-1, indeces[1]], dtype=np.int32)
	return return_indeces


@jit(nopython=True, nogil=True, cache=True)
def interpolate_linear(x, x0, x1, f0, f1):

	a=(f1-f0)/(x1-x0)
	b=f0-a*x0
	return a*x+b


@jit(nopython=True, nogil=True, cache=True)
def interpolate_field(p, X, Y, field_components):

	dimension=2
	c_p=closest_point_grid(p, X, Y)

	if np.all(c_p==np.array([-1,-1,-1,-1], dtype=np.int32)):
		return np.array([0.,0.], dtype=np.float64)
	elif dimension==2:
		xl=c_p[0]
		xu=c_p[1]
		yl=c_p[2]
		yu=c_p[3]

	lenx=len(X)
	leny=len(Y)    

	"""if dimension==3:
		Ex,Ey,Ez=field_components

		i=xl+lenx*yl+lenx*leny*zl; LLL=np.array([Ex[i],Ey[i],Ez[i]])
		i=xl+lenx*yl+lenx*leny*zu; LLU=np.array([Ex[i],Ey[i],Ez[i]])
		i=xl+lenx*yu+lenx*leny*zl; LUL=np.array([Ex[i],Ey[i],Ez[i]])
		i=xl+lenx*yu+lenx*leny*zu; LUU=np.array([Ex[i],Ey[i],Ez[i]])
		i=xu+lenx*yl+lenx*leny*zl; ULL=np.array([Ex[i],Ey[i],Ez[i]])
		i=xu+lenx*yl+lenx*leny*zu; ULU=np.array([Ex[i],Ey[i],Ez[i]])
		i=xu+lenx*yu+lenx*leny*zl; UUL=np.array([Ex[i],Ey[i],Ez[i]])
		i=xu+lenx*yu+lenx*leny*zu; UUU=np.array([Ex[i],Ey[i],Ez[i]])

		x1=coords[0][xl]; x2=coords[0][xu]
		y1=coords[1][yl]; y2=coords[1][yu]
		z1=coords[2][zl]; z2=coords[2][zu]

		C00=interpolate_linear(p[0],x1,x2,LLL,ULL)
		C01=interpolate_linear(p[0],x1,x2,LLU,ULU)
		C10=interpolate_linear(p[0],x1,x2,LUL,UUL)
		C11=interpolate_linear(p[0],x1,x2,LUU,UUU)

		C0=interpolate_linear(p[1],y1,y2,C00,C01)
		C1=interpolate_linear(p[1],y1,y2,C10,C11)

		C=interpolate_linear(p[2],z1,z2,C0,C1)"""

	if True:
		pass
		Ex=field_components[0]
		Ey=field_components[1]

		LL=np.array([Ex[lenx*yl+xl], Ey[lenx*yl+xl]], dtype=np.float64)
		UL=np.array([Ex[lenx*yu+xl], Ey[lenx*yu+xl]], dtype=np.float64)
		UR=np.array([Ex[lenx*yu+xu], Ey[lenx*yu+xu]], dtype=np.float64)
		LR=np.array([Ex[lenx*yl+xu], Ey[lenx*yl+xu]], dtype=np.float64)

		x1=X[xl]; x2=X[xu]
		y1=Y[yl]; y2=Y[yu]

		C0=interpolate_linear(p[0],x1,x2,LL,LR)
		C1=interpolate_linear(p[0],x1,x2,UL,UR)

		C=interpolate_linear(p[1],y1,y2,C0,C1)

	return C


@jit(nopython=True, nogil=True, cache=True)
def is_inside(p, edges):

	conditions=[p[0]<edges[0], p[0]>edges[1], p[1]<edges[2], p[1]>edges[3]]
	if len(p)==3:
		conditions.append(p[2]<edges[4])
		conditions.append(p[2]>edges[5])
	return not np.array(conditions).any()


@jit(nopython=True, nogil=True, cache=True)
def trajectory_line(p, X, Y, field_components, dn, edges, diffusion_on, units, print_point=False):

	length=0
	x_tmp=np.array([p[0]])
	y_tmp=np.array([p[1]])

	while is_inside(p, edges):
		E=interpolate_field(p, X, Y, field_components)
		normE=np.linalg.norm(E)
		dp=-dn*E/normE
		if diffusion_on:
			pass#dp=diffuse(dp, E, units)
			#length+=np.linalg.norm(dp)
		else:
			length+=dn
		p=p+dp
		x_tmp=np.concatenate((x_tmp,np.array([p[0]])))
		y_tmp=np.concatenate((y_tmp,np.array([p[1]])))
		if print_point:
			print(p)

	return x_tmp, y_tmp

