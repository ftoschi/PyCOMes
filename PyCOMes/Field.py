from .utils import *

class FieldNotFound(Exception):
    pass

class InvalidArgument(ValueError):
    pass

class Field():
    
    def __init__(self, file_name=None, skip_rows=9):
        
        self.file_name=file_name
        self._skip_rows=skip_rows
        
        if file_name==None:
            self.field=pd.DataFrame({})
            self.params={}
            self.vars={}
            self.dimension=0
            return
            
        if not isinstance(file_name, str):
            raise InvalidArgument
        
        self.params=read_params(file_name)
        self.vars=read_vars(file_name)
        self.dimension=read_dimension(file_name)
        self._load_txt()
        return
    
    def _load_txt(self):
        
        self.head=make_head(self.file_name)
        self.field=pd.read_csv(self.file_name, skiprows=self._skip_rows, delim_whitespace=True, header=None,
                               names=self.head, usecols=self.head)
        
        if self.dimension==2:
            try:
                self.X=np.unique(self.field.x)
                self.Y=np.unique(self.field.y)
            except:
                self.X=np.unique(self.field.r)
                self.Y=np.unique(self.field.z)
            
        if self.dimension==3:
            self.X=np.unique(self.field.x)
            self.Y=np.unique(self.field.y)
            self.Z=np.unique(self.field.z)
        
        return
    
    def load_txt(self,file_name,skip_rows=9,verbose=False):
        
        self.file_name=file_name
        self._skip_rows=skip_rows
        self._load_txt(file_name) 
        if verbose:
            self.file_header()
        return
    
    def file_header(self):
        
        with open(self.file_name,'r') as f:
            for i in range(self._skip_rows):
                print(f.readline())
        return

    def set_parameters(self, params: dict):
        
        if len(params)!=len(self.params):
            raise InvalidArgument("the number of parameters selected does not match the number of parameters needed.")
        if not all(p in self.params.keys() for p in params):
            raise InvalidArgument(f"unknow parameters. The accepted parameters are: {list(self.params.keys())}.")
        if not self.params:
            return
        self.selected_params=params
        self._selected_params_suffix=[]
        for i in self.params.keys():
            if not params[i] in self.params[i] and not params[i] in [str(p) for p in self.params[i]]:
                raise InvalidArgument(f"value not accepted for {i} parameter. The accepted values are {list(self.params[i])}.")
            self._selected_params_suffix.append(f'{i}={params[i]}')
        return
    
    def _contains_field(self):
        
        if self.dimension==2:
            return all(i in self.vars.keys() for i in ['Ex','Ey']) or all(i in self.vars.keys() for i in ['Er','Ez'])
        if self.dimension==3:
            return all(i in self.vars.keys() for i in ['Ex','Ey','Ez'])
    
    def get_field_components(self):
        
        if not self._contains_field():
            raise FieldNotFound
        e_field_components=['Ex', 'Ey'] if 'Ex' in self.vars.keys() else ['Er', 'Ez']
        if self.params==None:
            string_Ex=e_field_components[0]
            string_Ey=e_field_components[1]
        else:
            string_Ex=[i for i in self.head if all(k in i.split(' ') for k in [e_field_components[0]]+self._selected_params_suffix)][0]
            string_Ey=[i for i in self.head if all(k in i.split(' ') for k in [e_field_components[1]]+self._selected_params_suffix)][0]
        Ex=self.field[string_Ex]
        Ey=self.field[string_Ey]
        if self.dimension==3:
            e_field_components.append('Ez')
            if self.params == None:
                string_Ez = e_field_components[2]
            else:
                string_Ez=[i for i in self.head if
                           all(k in i.split(' ') for k in [e_field_components[2]]+self._selected_params_suffix)][0]
            Ez=self.field[string_Ez]
            return Ex, Ey, Ez
        return Ex, Ey
        
