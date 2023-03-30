from .utils import *
from .interpolation import *
import pandas as pd
import pickle


class FieldNotFound(Exception):
    pass


class InvalidArgument(ValueError):
    pass


class Field:

    def __init__(self, file_name=None, skip_rows=9):

        self.file_name = file_name
        self._skip_rows = skip_rows

        if file_name is None:
            self.field = pd.DataFrame({})
            self.params = {}
            self.vars = {}
            self.dimension = 0
            return

        self._path = os.path.dirname(self.file_name)
        if not isinstance(file_name, str):
            raise InvalidArgument('field should be given as an external file.')

        self.load_file()
        self.selected_params = None
        self._selected_params_suffix = None
        
        if self.params is None:
            self.E_components = self.get_field_components()
            
        return
    
    def __call__(self, p, params=None):
        
        if not params is None:
            if not isinstance(params, dict):
                raise InvalidArgument("the parameter selection should be a dictionary.")
            self.set_parameters(params)
                        
        if self.dimension == 2:
            return interpolate_field(np.asarray(p, dtype=np.float64),
                                     np.asarray(self.X, dtype=np.float64),
                                     np.asarray(self.Y, dtype=np.float64),
                                     np.asarray(self.E_components[0], dtype=np.float64),
                                     np.asarray(self.E_components[1], dtype=np.float64))
        else:
            return interpolate_field_3D(np.asarray(p, dtype=np.float64),
                                        np.asarray(self.X, dtype=np.float64),
                                        np.asarray(self.Y, dtype=np.float64),
                                        np.asarray(self.Z, dtype=np.float64),
                                        np.asarray(self.E_components[0], dtype=np.float64),
                                        np.asarray(self.E_components[1], dtype=np.float64),
                                        np.asarray(self.E_components[2], dtype=np.float64))

    def load_txt(self, file_name=None):

        if file_name is None:
            file_name = self.file_name
        else:
            self.file_name = file_name
        self.params = read_params_txt(self.file_name)
        self.vars = read_vars_txt(self.file_name)
        self.dimension = read_dimension_txt(self.file_name)
        self._load_txt()

    def _load_txt(self):

        self.head = make_head(self.file_name)
        self.field = pd.read_csv(self.file_name, skiprows=self._skip_rows, delim_whitespace=True, header=None,
                                 names=self.head, usecols=self.head)

        self._get_unique()

        return

    def _get_unique(self):

        if self.dimension == 2:
            try:
                self.X = np.unique(self.field.x)
                self.Y = np.unique(self.field.y)
            except:
                self.X = np.unique(self.field.r)
                self.Y = np.unique(self.field.z)

        if self.dimension == 3:
            self.X = np.unique(self.field.x)
            self.Y = np.unique(self.field.y)
            self.Z = np.unique(self.field.z)

    def set_parameters(self, params: dict):

        if len(params) != len(self.params):
            raise InvalidArgument("the number of parameters selected does not match the number of parameters needed.")
        if not all(p in self.params.keys() for p in params):
            raise InvalidArgument(f"unknown parameters. The accepted parameters are: {list(self.params.keys())}.")
        if not self.params:
            return
        self.selected_params = params
        self._selected_params_suffix = []
        for i in self.params.keys():
            if not str(params[i]) in [str(p) for p in self.params[i]]:
                raise InvalidArgument(
                    f"value not accepted for {i} parameter. The accepted values are {list(self.params[i])}.")
            self._selected_params_suffix.append(f'{i}={params[i]}')
        self.E_components = self.get_field_components()
        return

    def _contains_field(self):

        if self.dimension == 2:
            return all(i in self.vars.keys() for i in ['Ex', 'Ey']) or all(i in self.vars.keys() for i in ['Er', 'Ez'])
        if self.dimension == 3:
            return all(i in self.vars.keys() for i in ['Ex', 'Ey', 'Ez'])
        
    def get_field_components(self):

        if not self._contains_field():
            raise FieldNotFound
        e_field_components = ['Ex', 'Ey'] if 'Ex' in self.vars.keys() else ['Er', 'Ez']
        if self.params is None or len(self.params) == 0:
            string_ex = e_field_components[0]
            string_ey = e_field_components[1]
        elif len(self.params) != 0 and self.selected_params is None:
            raise NotImplementedError('parameter not selected.')
        else:
            string_ex = [i for i in self.head if
                         all(k in i.split(' ') for k in [e_field_components[0]] + self._selected_params_suffix)][0]
            string_ey = [i for i in self.head if
                         all(k in i.split(' ') for k in [e_field_components[1]] + self._selected_params_suffix)][0]
        Ex = self.field[string_ex]
        Ey = self.field[string_ey]
        if self.dimension == 3:
            e_field_components.append('Ez')
            if self.params is None:
                string_Ez = e_field_components[2]
            else:
                string_Ez = [i for i in self.head if
                             all(k in i.split(' ') for k in [e_field_components[2]] + self._selected_params_suffix)][0]
            Ez = self.field[string_Ez]
            return Ex, Ey, Ez
        return Ex, Ey

    def to_pickle(self, path=None, dtype=np.float32):

        if path is None:
            path = self._path
        path = path + '/' if not path.endswith('/') else path
        _file_name = self.file_name.split('/')[-1]

        output = dict(file_name=_file_name.split('.')[0] + '.pkl',
                      dimension=self.dimension,
                      params=self.params,
                      vars=self.vars,
                      head=self.head,
                      field=self.field.astype(dtype))
        with open(path + _file_name.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(output, f)
            f.close()

    def load_pickle(self, file_name=None):

        if not file_name is None:
            self.file_name = file_name
        with open(self.file_name, 'rb') as f:
            output = pickle.load(f)
            f.close()

        self.file_name = output['file_name']
        self.params = output['params']
        self.dimension = output['dimension']
        self.vars = output['vars']
        self.head = output['head']
        self.field = output['field']
        self._get_unique()
        
    def make_xyz(self, Xs, Ys, Zs, k=1):
      
      zs, ys, xs = np.meshgrid(Zs, Ys, Xs, indexing='ij')
      self.X = Xs
      self.Y = Ys
      self.Z = Zs
      
      if(hasattr(self, "field")):
        self.field['x'] = xs.ravel() * k
        self.field['y'] = ys.ravel() * k
        self.field['z'] = zs.ravel() * k
      else:
        self.field = pd.DataFrame({'x': xs.ravel() * k,
                                   'y': ys.ravel() * k,
                                   'z': zs.ravel() * k})
      return
        
    def load_npy(self, file_name=None):
    
      if not file_name is None:
        self.file_name = file_name
      
      # implemented only for 3D file map
      self.params = None
      self.vars = {'x': 'mm', 'y': 'mm', 'z': 'mm',
                   'Ex': 'V/cm', 'Ey': 'V/cm', 'Ez': 'V/cm'}
      self.dimension = 3
      self.head = 'DUMMY'
      field_tmp = np.load(file_name, allow_pickle=True)
      if(hasattr(self, "field")):
        self.field['Ex'] = field_tmp[0]
        self.field['Ey'] = field_tmp[1]
        self.field['Ez'] = field_tmp[2]
      else:
        self.field = pd.DataFrame({'Ex': field_tmp[0], 
                                   'Ey': field_tmp[1],
                                   'Ez': field_tmp[2]})
      return
        
    def load_vti(self, file_name=None):
      
      print("Not implemented. Bye bye.")
      return -1

    def load_file(self, file_name=None):

        if file_name is None:
            file_name = self.file_name
        extension = get_extension(self.file_name)
        loading_function = {FileType.TXT: self.load_txt,
                            FileType.PICKLE: self.load_pickle,
                            FileType.NPY: self.load_npy,
                            FileType.VTI: self.load_vti}

        load_function = loading_function[extension]
        load_function(file_name)
        
    def make_dummy(self, dimension, coordinates, components, unit_space='mm', unit_field='V/cm'):
        
        coordinates = np.array(coordinates, dtype=np.float64)
        components = np.array(components, dtype=np.float64)
        
        assert coordinates.shape[0] == dimension, f'coordinate shape {coordinates.shape[0]} and dimension {dimension} do not match'
        assert components.shape[0] == dimension, f'components shape {components.shape[0]} and dimension {dimension} do not match'
        
        self.file_name = 'dummy'
        self.params = {}
        self.dimension = dimension
        if dimension == 2:
            self.vars = {'x': unit_space, 'y': unit_space, 
                         'Ex': unit_field, 'Ey': unit_field}
            field_tmp = pd.DataFrame({'x': np.asarray(coordinates[0], dtype=np.float64),
                                      'y': np.asarray(coordinates[1], dtype=np.float64),
                                      'Ex': np.asarray(components[0], dtype=np.float64),
                                      'Ey': np.asarray(components[1], dtype=np.float64)})
        else:
            self.vars = {'x': unit_space, 'y': unit_space, 'z': unit_space,
                         'Ex': unit_field, 'Ey': unit_field, 'Ez': unit_field}
            field_tmp = pd.DataFrame({'x': np.asarray(coordinates[0], dtype=np.float64),
                                      'y': np.asarray(coordinates[1], dtype=np.float64),
                                      'z': np.asarray(coordinates[2], dtype=np.float64),
                                      'Ex': np.asarray(components[0], dtype=np.float64),
                                      'Ey': np.asarray(components[1], dtype=np.float64),
                                      'Ez': np.asarray(components[2], dtype=np.float64)})
        self.field = field_tmp
        self._get_unique()
        self.E_components = self.get_field_components()