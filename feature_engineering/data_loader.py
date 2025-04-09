import numpy as np
import h5py
from math import ceil, floor


class DarusLoader():
  """
  Create an object which docks to the available hdf5 dataset and loads
  the data thereof. The object can then be simply indexed and the requested
  data, specified per 'switches', will be returned as tuple in the fixed
  order:
  images, targets = DarusLoader[20:1000:3]
  each data kind can be explicitely loaded via getter methods.


  Examples:
  ---------
    load = DarusLoader( set_type='bench', images=False, phase_contrast=contrasts)
    kappa = load[:] #load the target values
    load.load_targets(False)
    load.load_features(True)
    features = load[:] #only load the features
    load.close()

    n_train = 1500
    loader2 = DarusLoader( set_type='train', features=True, imgs=True ) #does accept slighty flexible inputs
    images, kappa, features = DarusLoader[ 500:200:-1] #accepts negative increments
    training_samples = np.random.permutation( np.arange( n_train) )
    img_train, k_train, fts_train = DarusLoader[ training_samples] #accepts arbitrary arrays/lists
  """
  def __init__( self, set_type='train', data_path='./', phase_contrast=5,**load):
    """
    Initialize all path dependencies and specify default return values.
    Parameters:
    -----------
    set_type:       str, default 'train'
                    choose between 'train' and 'benchmark', specifies which
                    dataset to load, also sets the number of samples available
                    as well as the available phase contrasts
    data_path:      str, default './'
                    path to the hdf5 file, looks for 'feature_engineering_data.h5'
                    If a hdf5 file ending is given the provided filename is taken instead
    phase_contrast: int or list of ints, default 5
                    of which phase contrast to load the data. If a list of
                    ints is given it will affect the feature loading,
                    inserting one more feature being the phase contrast.
    **load          kwargs with default arguments
                    what data to load, the kwargs do not have to be identically
                    specified, it suffices if the substring is in the kwarg
      image:        bool, default True
      target/kappa: bool, default True
      feature:      bool, default False
    """
    ##input preprocessing
    load_images = True #default arguments
    load_targets = True
    load_features = False
    if (key := np.argwhere( [ 'image' in x.lower() for x in load.keys()]) ).size > 0 or (
        (key := np.argwhere( [ 'img' in x.lower() for x in load.keys()]) ).size  > 0):
        load_images = load.pop( list(load.keys())[key.squeeze()] )
    if (key := np.argwhere( [ 'feature' in x.lower() for x in load.keys()]) ).size > 0 or(
       (key := np.argwhere( [ 'fts' in x.lower() for x in load.keys()]) ).size > 0)     :
        load_features = load.pop( list(load.keys())[key.squeeze()] )
    if (key := np.argwhere( [ 'kappa'  in x.lower() for x in load.keys()]) ).size > 0 or (
       (key := np.argwhere( [ 'target' in x.lower() for x in load.keys()]) ).size > 0 ):
        load_targets = load.pop( list(load.keys())[key.squeeze()] )
    if '.h5' in data_path[-3:] or '.hdf5' in data_path[-5:]:
        self.filename = data_path
    else:
        data_path =  data_path + '/' if '/' != data_path [-1] else data_path
        self.filename = data_path + 'data/feature_engineering_data.h5'
    ## preallocation of internal variables for loading
    self.open()
    self.image_switch = load_images #reallocation of variables due to cross referncing
    self.image_formatting = 'tf'
    self.target_formatting = 'mandel'
    self.target_switch = load_targets
    self.feature_switch = load_features
    self.phase_contrast = [phase_contrast] if not hasattr( phase_contrast, '__iter__') else phase_contrast 
    self.set_type( set_type)
    self.set_phase_contrast( phase_contrast )
    ## other useful variable allocations
    self.n_samples = len(self)
    self.__name__  = 'DarusLoader' #for console feedback below

  def close( self):
      """
      Close the reference to file such that it can be opened by
      other sources
      """
      self.h5file.close()

  def open( self):
      """ reopen the access to file if it has been previously closed"""
      self.h5file = h5py.File( self.filename, 'r' )


  ##### Variable allocation for all loading switches
  def set_phase_contrast( self, phase_contrast):
      """
      specify the phase contast to load in the target values and
      possibly in the features with the extra feature
      Parameters:
      -----------
      phase_contrast: list of ints or int, default [5]
                    which phase contrasts to load. Both sets have 
                    different available, for benchmark there is
                    range( 2,101) and for train set it is [2,5,10,20,50,100]
      """
      self.phase_contrast = [phase_contrast] if not hasattr( phase_contrast, '__iter__') else phase_contrast
      self.extra_feature = len(self.phase_contrast) > 1
      self.load_targets( self.target_switch, self.target_formatting)

  def set_type( self, set_type='train' ):
      """
      specify from which dataset to load the data
      Takes 'train' and 'benchmark' as inputs
      """
      self._set_type    = 'train' if 'train' in set_type.lower() else 'benchmark'
      self.image_path   = '{}_set/image_data'.format( self._set_type)
      self.target_path  = '{}_set/effective_heat_conductivity/contrast_{{}}'.format( self._set_type)
      self.feature_path = '{}_set/feature_vector'.format( self._set_type)
      ### redo the switches to adjust the containers for loading correspondingly
      self.load_images( self.image_switch, self.image_formatting)
      self.load_targets( self.target_switch, self.target_formatting ) #reinvoke the containers
      self.load_features( self.feature_switch)

  def load_images( self, switch=True, formatting='tensorflow'):
    """
    Specify if the images should be loaded via a boolean <switch>
    Parameters:
    -----------
    switch:     bool, default True
                whether or not to load the images
    formatting: str, default 'tensorflow'
                how to format the images, i.e. if the 1 channel should
                be added (required for machine learning) by choosing 
                'tensorflow' or 'tf', all other inputs give the flat 
                image representation with (n_samples,400,400)
    Returns:
    --------
    None:       the data is returned in the get_images() method, 
                or via indexing
    """
    self.image_switch = switch
    self.image_formatting = formatting
    if self.image_formatting.lower() in ['tf', 'tensorflow']:
        self.image_format = lambda images: images.reshape( -1, 400, 400, 1)
    else:
        self.image_format = lambda images: images
    self.image_container  = self.h5file[ self.image_path]

  def load_targets( self, switch=True, formatting='mandel'):
    """ 
    Specify if the targets should be loadded and which targets to load
    upon indexing.
    Parameters:
    -----------
    switch:     bool, default True
                activation switch for loading
    formatting:     string, default 'mandel'
                    how to arrange the output values, always given with
                    the number of samples is always givne in the first index
                    choose between 'mandel', 'voigt', and 'array/matrix'
    Returns:
    --------
    None:           only sets internal variables
    """
    self.target_container = [ self.h5file[self.target_path.format( x)] for x in self.phase_contrast ]
    self.target_switch = switch
    self.target_formatting = formatting
    if self.target_formatting.lower() in ['mandel', 'default']:
        self.target_format = lambda x: x
        self.target_shape  = (3,)
    elif self.target_formatting.lower() == 'voigt':
        self.target_format = lambda x: np.vstack( [x[:,0], x[:,1], (2/2**0.5)*x[0,1] ])
        self.target_shape  = (3,)
    elif self.target_formatting.lower() in ['array', 'matrix']:
        self.target_format = lambda x: np.stack(
        [np.stack( [x[:,0], 1/2**0.5*x[:,2]],axis=1),
         np.stack( [1/2**0.5*x[:,2], x[:,1]],axis=1) ]
        , axis=2 )
        self.target_shape  = (2,2)

  def load_features(self, switch=True):
    """Specify if the features should be loaded via a boolean <switch>"""
    self.feature_switch = switch
    self.feature_container = self.h5file[ self.feature_path]



  ##### getter methods of explicit return statements
  def get_images( self, idx):
    """
    return the image data of requested indices. Takes in any
    default indexing objects, see idx_to_array for reference
    """
    idx = self.idx_to_array( idx)
    images = self.chunked_loading( idx, self.image_container)
    return self.image_format( images)

  def get_targets( self, idx):
    """
    return the target values of requested indices. Takes in any
    default indexing objects, see idx_to_array for reference
    """
    idx = self.idx_to_array( idx)
    targets = []
    for container in self.target_container:
        targets.append( self.chunked_loading( idx, container)  )
        targets[-1] = self.target_format( targets[-1])
    if len( targets) == 1:
        return targets[0]
    else:
        return targets


  def get_features( self, idx):
    """
    return the target values of requested indices. Takes in any
    default indexing objects, see idx_to_array for reference
    """
    idx = self.idx_to_array( idx)
    features = self.chunked_loading( idx, self.feature_container)
    if len( self.phase_contrast) > 1:
        multi_contrast = []
        for contrast in self.phase_contrast:
            multi_contrast.append( np.insert( features, 1, contrast, axis=1)  )
        features = multi_contrast
    return features


  def get_basis( self, n_xi=13):
    """
    Return the  reduced basis in memory. The size of the basis can
    be specified to obtain only as much reduced coefficients as desired
    on projection. The maximum number of n_xi is 259.
    Parameters:
    -----------
    n_xi:     int, default 13
              number of eigenmodes to return, set to None to get all
    Returns:
    --------
    rb:       numpy nd-array
              reduced basis of shape (400**2, n_xi)
    """
    return self.h5file['feature_extractors/pcf_basis'][:,:n_xi]


## required functions for loading
  def idx_to_array( self, idx):
    """
    convert any common indexing method into a numpy array of integers
    This method may also be used as function
    """
    if isinstance( idx, slice):
        if idx.start is None: start = 0
        elif idx.start < 0: start = self.n_samples + idx.start
        else: start = idx.start
        if idx.step is None: step = 1
        else: step = idx.step
        if idx.stop is None: stop = self.n_samples
        elif idx.stop < 0: stop = self.n_samples + idx.stop
        else: stop = idx.stop
        idx        = list( range( start, stop, step) )
    elif isinstance( idx, int):
        idx = [idx]
    elif isinstance( idx, (list,tuple,np.ndarray)):
        if isinstance( idx, np.ndarray):
          idx = idx.squeeze()
          if idx.ndim != 1:
            raise Exception( 'only 1d-arrays are admissible for indexing {}'.format( self.__name__ ) )
    else:
        raise Exception( 'Illegal indexing for {}, accepted are int/slice/list/tuple/np.1darray'.format( self.__name__))
    return np.array( idx )


  def chunked_loading(self, idx, data_container ):
    """
    Knowing the chunks of the image dataset it is faster to manually load
    each chunk separately and concatenate it afterward. This method does
    exactly that. Basically returns all requested datasets in idx. It
    assumes that the datasets are given in tensorflow notation with samples
    in the first index, and that is where chunks are laid out, always
    taking full samples. No performance guarantees can be made for
    differently chunked data.
    This method may also be used as function.
    Parameters:
    -----------
    idx:            list like of ints
                    requested indices of the dataset
    data_container: path do h5py.dataset
                    index acessible reference to the dataset
    """
    ## input preprocessing
    chunksize = data_container.chunks[0]
    sorting = np.argsort( idx ).squeeze()
    idx = idx[sorting]
    n_chunks = ceil((idx[-1]-idx[0])/chunksize )
    data = []
    i = 0
    j = 0
    while i < len( idx ):
        chunk_index = [idx[i]]
        current_chunk = idx[i] // chunksize
        i += 1 #adding first index
        while i < len(idx) and (idx[i] // chunksize) == current_chunk:
            chunk_index.append( idx[i] ) #indices of current chunk
            i += 1 #added another index
        data.append( data_container[chunk_index] ) #load the single chunk
        j+= 1
    data = np.concatenate( data, axis=0 )
    return data[np.argsort(sorting)] #undo sorting to requested order


  ##### dunder methods for indexing etc
  def __getitem__( self, idx):
    """
    return the data specified with the previous switches. Takes in any
    default indexing, see 'idx_to_array' or a possible error measage for
    a detailed description.
    Parameters:
    -----------
    idx:    slice-like objects
            specified index
    Returns:
    --------
    data:   list of numpy nd-array or numpy nd-array
            requested data
    """
    data = []
    if self.image_switch:
        data.append( self.get_images( idx) )
    if self.target_switch:
        data.append( self.get_targets( idx) )
    if self.feature_switch:
        data.append( self.get_features( idx) )
    if len( data) == 1:
        return data[0]
    else:
        return data


  def __len__( self):
      """ Return the number of samples in the corresponding set """
      n_samples = 30000 if self._set_type == 'train' else 1500
      return n_samples

  #### alias and function shadows
  def load_image(self, *args, **kwargs):
      """ alias for load_images, see its documentation """
      return self.load_images( *args, **kwargs )
  def load_kappa(self, *args, **kwargs):
      """ alias for load_targets, see its documentation """
      return self.load_targets( *args, **kwargs )
  def load_target(self, *args, **kwargs):
      """ alias for load_targets, see its documentation """
      return self.load_targets( *args, **kwargs )
  def load_kappas(self, *args, **kwargs):
      """ alias for load_targets, see its documentation """
      return self.load_targets( *args, **kwargs )
  def load_feature(self, *args, **kwargs):
      """ alias for load_features, see its documentation """
      return self.load_features( *args, **kwargs )
  def get_bases( self, *args, **kwargs):
      """ alias for get_basis, see its documentation """
      return self.get_basis( *args, **kwargs )
