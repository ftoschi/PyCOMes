from numba import jit
import numpy as np
from numba.targets.rangeobj import range_iter_len

OUT_OF_EDGES=-1

@jit(cache=True)
def closest_point_grid(p, coords):
	               
	indeces=[]
	indeces.append(np.digitize(p[0], coords[0]))
	indeces.append(np.digitize(p[1], coords[1]))
	if len(coords)==3:
		indeces.append(np.digitize(p[2], coords[2]))
	indeces=np.array(indeces)
        
	length=np.array([len(coords[0]), len(coords[1])]) if len(coords)==2 else np.array([len(coords[0]), len(coords[1]), len(coords[2])])
	for i in indeces:
		if i==0: 
			return OUT_OF_EDGES
	for i in range(len(indeces)):
		if indeces[i]==length[i]:
			return OUT_OF_EDGES
        
	return_indeces=[indeces[0]-1, indeces[0], indeces[1]-1, indeces[1]]
	if len(coords)==3:
		return_indeces.append([indeces[2]-1, indeces[2]])
	return return_indeces

    
@jit(nopython=True, nogil=True, cache=True)
def interpolate_linear(x, x0, x1, f0, f1):
    
	a=(f1-f0)/(x1-x0)
	b=f0-a*x0
	return a*x+b


@jit(cache=True)
def interpolate_field(p, coords, field_components):
        
	dimension=len(coords)
	c_p=closest_point_grid(p, coords)

	if c_p==OUT_OF_EDGES:
		return c_p
	elif dimension==2:
		xl,xu,yl,yu=c_p
	elif dimension==3:
		xl,xu,yl,yu,zl,zu=c_p

	lenx=len(coords[0])
	leny=len(coords[1])
	if dimension==3:
		lenz=len(coords[2])
            
	if dimension==3:
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
    
		C=interpolate_linear(p[2],z1,z2,C0,C1)  
        
	elif dimension==2:
		Ex,Ey=field_components

		LL=np.array([Ex[lenx*yl+xl],Ey[lenx*yl+xl]]) 
		UL=np.array([Ex[lenx*yu+xl],Ey[lenx*yu+xl]])
		UR=np.array([Ex[lenx*yu+xu],Ey[lenx*yu+xu]])
		LR=np.array([Ex[lenx*yl+xu],Ey[lenx*yl+xu]])
            
		x1=coords[0][xl]; x2=coords[0][xu]
		y1=coords[1][yl]; y2=coords[1][yu]

		C0=interpolate_linear(p[0],x1,x2,LL,LR)
		C1=interpolate_linear(p[0],x1,x2,UL,UR)
    
		C=interpolate_linear(p[1],y1,y2,C0,C1)
        
	return C


@jit(cache=True)
def is_inside(p, edges):
        
	conditions=[p[0]<edges[0], p[0]>edges[1], p[1]<edges[2], p[1]>edges[3]]
	if len(p)==3:
		conditions.append(p[2]<edges[4])
		conditions.append(p[2]>edges[5])
	return not np.array(conditions).any()


@jit(cache=True)
def trajectory_line(p, coords, field_components, dn, edges, plot=False, print_point=False):
        
	length=0
	x_tmp=[p[0]]
	y_tmp=[p[1]]
		
	while is_inside(p, edges):
		E=np.array(interpolate_field(p, coords, field_components))
		dp=-dn*E/np.linalg.norm(E)
		length+=dn
		p=p+dp
		if plot:
			x_tmp.append(p[0])
			y_tmp.append(p[1])
		if print_point:
			print(p)

	if plot:
		return length, np.array(x_tmp), np.array(y_tmp)
	return length
