import numpy as np
from enum import Enum, unique
import os
import re


@unique
class FileType(Enum):
    TXT = 0
    PICKLE = 1
    NPY = 2


class WrongDimension(Exception):
    pass


def num(s):
    try:
        k = int(s)
    except:
        k = float(s)
    return k


def get_extension(file_name):

    extension = os.path.splitext(file_name)[1][1:]
    known_extension = {'text': ['txt', 'csv'], 'pickle': ['pkl', 'pickle', 'p'], 'numpy': ['npy']}
    if extension in known_extension['text']:
        return FileType.TXT
    elif extension in known_extension['pickle']:
        return FileType.PICKLE
    elif extension in known_extension['numpy']:
        return FileType.NPY
    else:
        raise TypeError(f"file type not recognized, known extensions are {known_extension}.")


def read_params_txt(file, skip_rows=8):
    """
        Returning the sweep parameters for a COMSOL simulation exported in a text file.
        
        - file: name of the text file;
        - skip_rows: rows to skip with data regarding the simulation.
    """

    with open(file, 'r') as f:
        for l in f:
            if not l.startswith('%'):
                break
            last_line = l
        f.close()
    header_elements = re.split('\s{1,100}|,', last_line)
    reg = re.compile('.*=.*')
    param_names = tuple(np.unique([i.split('=')[0] for i in list(filter(reg.match, header_elements))]))
    if not len(param_names):
        return None

    params = {}
    for p in param_names:
        params[p] = np.unique([l.split('=')[-1] for l in header_elements if l.startswith(p+'=')])

    return params


def make_head(file, skip_rows=8):
    """
    Returning the head to use when loading the field from the COMSOL output text file.
    
    - file: name of the text file;
    - skip_rows: rows to skip with data regarding the simulation.
    """

    with open(file, 'r') as f:
        for i in range(skip_rows):
            line = f.readline()
            if 'Dimension' in line:
                dim = int(line.split()[-1])
        head = f.readline()
        f.close()
    fields_params = read_params_txt(file, skip_rows)
    var = [v for v in read_vars_txt(file, skip_rows).keys()]
    if fields_params is None:
        return var
    head = head.split(' ')
    head = np.array([h.replace(',', '') for h in head if '=' in h])
    fields = read_params_txt(file, skip_rows).keys()
    head = np.reshape(head, (len(head) // len(fields), len(fields)))
    skip = len(var) - dim if var else dim
    head = head[::skip]
    head_var = var[:dim]
    for h in head:
        for e in var[dim:]:
            head_var.append(e + ' ' + ' '.join(h).replace('\n', ''))
    return head_var


def read_vars_txt(file, skip_rows=8):
    """
    Returning the variables stored in the input txt file from COMSOL (e.g. x, y, z, normE, etc.).
    
    - file: name of the text file;
    - skip_rows: rows to skip with data regarding the simulation.
    """

    variables = {}
    with open(file, 'r') as f:
        for line in f:
            if 'Length unit' in line:
                len_unit = line.split()[-1]
            if 'Dimension' in line:
                dim = int(line.split()[-1])
            if not line.startswith('%'):
                break
            last_line = line
        f.close()

    header_elements = re.split('\s{1,100}|,', last_line)
    coords = header_elements[1:dim+1]

    var_temp = np.unique(re.findall('\w*\s?\(.*?\)', last_line))
    variables = {c: len_unit for c in coords}
    variables.update({re.split(' \(', l)[0]: re.split('\(|\)', l)[1] for l in var_temp})

    return variables


def read_dimension_txt(file):
    """
    Returning the dimension of the COMSOL simulation.
    
    - file: name of the text file.
    """

    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if 'Dimension' in line:
                dim = int(line.split()[-1])
                f.close()
                break
    return dim