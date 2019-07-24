import numpy as np
from numba import jit, jitclass
import pandas as pd

def num(s):
    try:
        k=int(s)
    except:
        k=float(s)
    return k

class WrongDimension(Exception):
    pass

def read_params(file, skip_rows=8):

    """
        Returning the sweep parameters for a COMSOL simulation exported in a text file.
        
        - file: name of the text file;
        - skip_rows: rows to skip with data regarding the simulation.
    """
        
    with open(file,'r') as f:
        for i in range(skip_rows):
            f.readline()
        head=f.readline()
        f.close()
    head=np.unique(head.split(' '))
    head=[h.replace(',','') for h in head if '=' in h]
    if not len(head):
        return None
    head_field=[]
    head_value={}
    for h in head:
        h.replace('/n', '')
        field,value=h.split('=')
        head_field.append(field)
        if field not in head_value:
            head_value[field]=[num(value)]
        else:
            if num(value) in head_value[field]:
                continue
            head_value[field].append(num(value))
    return head_value

def make_head(file,skip_rows=8):
    
    """
    Returning the head to use when loading the field from the COMSOL output text file.
    
    - file: name of the text file;
    - skip_rows: rows to skip with data regarding the simulation.
    """

    with open(file,'r') as f:
        for i in range(skip_rows):
            line=f.readline()
            if 'Dimension' in line:
                dim=int(line.split()[-1])
        head=f.readline()
        f.close()
    fields_params=read_params(file,skip_rows)
    var=[v for v in read_vars(file,skip_rows).keys()]
    if fields_params==None:
        return var
    head=head.split(' ')
    head=np.array([h.replace(',','') for h in head if '=' in h])
    fields=read_params(file,skip_rows).keys()
    head=np.reshape(head, (len(head)//len(fields), len(fields)))
    skip=len(var)-dim if var else dim
    head=head[::skip]
    HEAD=var[:dim]
    for h in head:
        for e in var[dim:]:
            HEAD.append(e+' '+' '.join(h).replace('\n',''))
    return HEAD

def read_vars(file,skip_rows=8):
    
    """
    Returning the variables stored in the input txt file from COMSOL (e.g. x, y, z, normE, etc.).
    
    - file: name of the text file;
    - skip_rows: rows to skip with data regarding the simulation.
    """
    
    variables={}
    with open(file,'r') as f:
        for i in range(skip_rows):
            line=f.readline()
            if 'Length unit' in line:
                len_unit=line.split()[-1]
            if 'Dimension' in line:
                dim=int(line.split()[-1])
        head=f.readline()
        f.close()
    head=head.split()
    head=[h for h in head if h not in ['%','@'] and '=' not in h and ':' not in h]
    coordinates=head[:dim]
    for c in coordinates:
        variables[c]=len_unit
    for i in range(0,len(head[dim:])):
        if '.' not in head[dim:][i]:
            continue
        v=head[dim:][i].split('.')[-1]
        if v in variables:
            break
        else:
            unit=head[dim:][i+1].replace(')','')
            unit=unit.replace('(','')
            variables[v]=unit
    return variables

def read_dimension(file):
    
    """
    Returning the dimension of the COMSOL simulation.
    
    - file: name of the text file.
    """
    
    with open(file,'r') as f:
        while True:
            line=f.readline()
            if 'Dimension' in line:
                dim=int(line.split()[-1])
                f.close()
                break
    return dim
