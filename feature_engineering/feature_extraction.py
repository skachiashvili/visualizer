import numpy as np
from numpy import pi
from numpy.fft import fftn, ifftn
from scipy.stats import skew
from math import ceil, floor

import sys
sys.path.append('./feature_engineering/')
from data_loader import DarusLoader
from general_functions import tic, toc



def reduced_coefficients( images, n_xi=13):
    """
    Compute the reduced coefficients using the precomputed reduced basis
    found in the hdf5 file attached to the repository.
    The resolution of the images has to match the original resolution
    of the reduced basis, i.e. 400x400.
    CARE: this function allocates a full dense float array of image shape
    Parameters:
    -----------
    images:     numpy nd-array
                images of shape n_samples, 400, 400)
    n_xi:       int, default 13
                number of reduced coefficients to obtain, max is 259
    Returns:
    --------
    vol+xi:     numpy nd-arary
                feature vector of shape (n_samples, 1+n_xi)
                does also return the volume fraction in first index
    """
    ## input processing and data loading
    assert images.ndim >= 3, 'image data of size (n_samples, n_x, n_y) is required' 
    resolution = list( images.shape[1:] )
    if 1 in resolution:
        resolution = resolution.pop( resolution.index( 1) )
    n_voxels    = np.prod( resolution )
    n_samples   = images.shape[0]
    vol         = images.mean(1).mean(1)
    pcf_scaling = (vol * n_voxels**2 )[:,None,None]
    basis       = DarusLoader().get_basis()
    for i in range( n_xi ):
        basis[:,i] = fftn( basis[:,i].reshape( resolution) ).flatten().real
    basis[0,:] = 0 #enforce 0 mean below in projection 
    ## compute pcf of images (only in fourier space for computational speedup
    fourier = np.zeros( (n_samples, *resolution), dtype=complex )
    for i in range( n_samples):
        fourier[i] = fftn( images[i] ) 
    fourier = (fourier*np.conj(fourier)/pcf_scaling ) #rescaling
    ## feature computation
    feature_vector = [ vol[:,None]]
    feature_vector.append( ( fourier).reshape( n_samples, -1) @ basis )
    return np.concatenate( feature_vector, axis=1)


def band_features( images, width=4, angles=None):
    """ 
    get the peak value of a band diagonal to find the max 'directly diagonal'
    of the microstructure. Detects the presence and absence of the searched
    direction. Gives 8*2 features with default arguments.
    Parameters:
    -----------
    images:     numpy nd-array
                array of n flattened images of shape (n_samples, n_x, n_y)
    width:      int, default 2
                width of the diagonal bands (computes to 1+2*<width>
    angles:     list like of floats, default None
                at which angles to set the band diagonals, 0° is identity
                matrix. At default it computes 8 directions as pi/8 increments
    resolution: list like of ints
                original resolution of each flattened image
    Returns:
    --------
    features:   numpy nd-array
                feature vector of shape [n_samples, (2*len(angles), 2)]
    """
    ## input preprocessing and filter allocation
    if angles is None:
        angles = np.arange( 0, pi, pi/8) 
    n_angles  = len( angles)
    n_images  = images.shape[0]
    resolution = list( images.shape[1:] )
    if 1 in resolution:
        resolution = resolution.pop( resolution.index( 1) )
    n_voxels  = np.prod( resolution)
    features  = np.zeros( ( n_angles*2, n_images ) )
    diagonals = [ fftn( tilted_diagonal( resolution[0], width=width, angle=angle) ) for angle in angles]
    ## feature computation
    for i in range( n_images):
        frequency_spectrum = fftn( images[i] )
        for j in range( n_angles):
            feature            = ifftn( frequency_spectrum * diagonals[j]).real
            features[2*j,i ]   = feature.max() 
            features[2*j+1, i] = (1-feature.min()) 
    return features.T



def projected_wall( images ):
    """
    Motivation: Edges found the projection + notches of the image in the 
    edge direction (no matter the edge filter)
    Here it detects the projection of inclusion in the specified axis
    ommitting the notches. Gives 2 features.
    Parameters:
    -----------
    images:     numpy nd-array
                images of shape n_samples, n_x, n_y)
    Returns:
    --------
    features:   numpy nd-array
                features of each images of the projected length in each
                axis of shape (n_samples, 2)
    """
    n_images      = images.shape[0]
    features      = np.zeros( (n_images, 2) )
    features[:,0] = np.mean( np.minimum( images.sum(1), 1), 1 )
    features[:,1] = np.mean( np.minimum( images.sum(2), 1), 1 )
    return features





def vol_distribution( images, fullness=None, window=(50,50)): 
    """
    Counts the number of grids which have the local volume fraction 
    in the interval <fullness>. The grid is defined by <window<, if
    <window> is not a dividor of resolution then the last entry will
    contain a slight overlap due to periodicity. 
    Gives len( fullness)+2 features, default 7.
    Also computes the std and skewness of the volume fraction 
    distribution.
    Parameters:
    -----------
    images:     numpy nd-array
                array of n flattened images of shape (res, n)
    fullness:   list of floats, default None
                what local volume fraction should be counted, does also
                allow for multiple checks at once if 'fullness' is a list
                of lists of floats, 
                defaults to [(0,0), (0.0001,1/3), (1/3,2/3), (2/3, 0.999), (1,1)]
    window:     tuple of ints, default (50,50)
                size of the local neighbourhood
    Returns:
    --------
    n_filled    numpy nd-array
                number of cells which have the required volume fraction
                does normalize i.e. n_filled/n_cells of shape (n, 7) with
                default arguments
    """
    ##input preprocessing
    if fullness is None:
        epsilon  = 0.99999999999999/np.prod( images.shape[1:] )
        fullness = list( zip( np.arange(epsilon, 1.0, 1/3), np.minimum( 1-epsilon, np.arange( 1/3, 4/3, 1/3)) ) )
        fullness.append( [1,1] )
        fullness.insert( 0, [0,0] )
    window   = np.array( window).astype(int)
    n_cells  = np.prod( np.ceil( np.array( images.shape[1:]) / window ) )
    n_images = images.shape[0]

    # if an integer value has to be found
    if isinstance( fullness[0], float ) or isinstance( fullness[0], int ):
        n_filled = np.zeros( n_images) 
        for i in range( n_images):
            volume_fractions = average_pooling( images[i], window )
            n_filled[i]      = ((fullness[0] < volume_fractions ) * (volume_fractions < fullness[1] ) ).sum()
        n_filled = n_filled/n_cells
    ## else count the number of occurences inside the interval
    elif isinstance( fullness[0], list) or isinstance( fullness[0], tuple):
        n_full   = len( fullness)
        n_filled = np.zeros( (n_images, n_full+2) ) 
        for i in range( n_images):
            volume_fractions = average_pooling( images[i] , window)
            for j in range( n_full):
                if np.allclose( fullness[j], [1,1] ) or np.allclose( fullness[j], [0,0] ): #if its basically 0 or 1
                    n_filled[i,j] = np.isclose(volume_fractions, fullness[j][0]).sum() 
                else: 
                    n_filled[i,j] = ((fullness[j][0] <= volume_fractions ) * (volume_fractions <= fullness[j][1] ) ).sum() 
            n_filled[i,-2] = np.std( volume_fractions) 
            n_filled[i,-1] = skew( volume_fractions, axis=None) 
        n_filled[:,:-2] = n_filled[:,:-2]/ np.prod( volume_fractions.shape)
    else:
        raise ValueError( 'wrong input for "feature_extraction.vol_distribution" in <fullness>')
    return n_filled


def edge_distributions( images, window=(50,50)): 
    """ 
    Computes the average amount of edges in each window. Edges are detected
    in horizontal, vertical and diagonal edges. This yields a distribution
    of edges over the images and the mean, std and skewness of that distri-
    bution are taken.
    Parameters:
    -----------
    images:     numpy nd-array
                array of n flattened images of shape (res, n)
    window:     tuple of ints, default (50,50)
                size of the local neighbourhood
    resolution: tuple of ints, default (400,400)
                resolution/shape of a single image
    Returns:
    --------
    edge_features:  numpy 2d-array
                    mean, std and skewness in the distribution given by 
                    the cells of each of the edge detectors, shape (n_samples, 12)
    """
    ## input processing
    n_images   = images.shape[0]
    resolution = images.shape[1:]
    if 1 in resolution:
        resolution = resolution.pop( resolution.index( 1) )
    #kernel allocation
    kernels    = []
    sobel_h = np.array( [1,0,-1] ).reshape(1,3)
    kernels.append( sobel_h)
    sobel_v = np.array( [1,0,-1] ).reshape(3,1)
    kernels.append( sobel_v)
    diagonal = np.array( [ [0, 0.5, 1],  [-0.5, 0, 0.5], [-1,-0.5,0] ] )
    kernels.append( diagonal)
    kernels.append( np.flip(diagonal, axis=0) )
    kernels  = [ kernel/ np.abs( kernel).sum() for kernel in kernels]
    kernels  = [ fftn( embed_kernel( kernel, resolution)) for kernel in kernels]
    features = np.zeros( (n_images, 3*len( kernels) ) )

    window  = np.array( window).astype(int)
    for i in range( n_images):
        frequency_spectrum = fftn( images[i].reshape( resolution) )
        for j in range( len( kernels)):
            edges             = np.abs( ifftn( frequency_spectrum * kernels[j]).real )
            feature_grid      = average_pooling( edges, window)
            features[i,3*j]   = np.mean( feature_grid) 
            features[i,3*j+1] = np.std( feature_grid) 
            features[i,3*j+2] = skew( feature_grid, axis=None) 
    return features


def full_computation( images, phase_contrasts=[5]):
    """
    compute all features from scratch by taking the FFT of the image
    only once. This function did hardcopy parts of the code of prior
    functions in the favor of computational efficency.
    The given parameters are hardwired and for now intended to be 
    used in the uploaded repository.
    Parameters:
    -----------
    images:     numpy nd-array
                image data of shape (n_samples, 400, 400)
    phase_contrasts:    list of ints, default [5]
                        which phase contrasts are considered
                        if multiple integers are given then a feature
                        is added at the second index.
    Returns:
    --------
    feature_vector:     numpy nd-array or list of numpy nd-arrays
                        feature vector of shape (n_samples, 51)
                        returns a list of (n_samples,52) feature vectors
                        if <len(phase)contrast >= 2> 
    """
    ##input procesing
    n_samples = images.shape[0]
    ### hardwired variables 
    resolution = images.shape[1:]
    n_voxels   = np.prod( resolution )
    n_xi       = 13
    # for band features
    width     = 4
    n_angles  = 8
    angles    = np.arange( 0, pi, pi/n_angles)
    diagonals = [ fftn( tilted_diagonal( resolution[0], width=width, angle=angle) ) for angle in angles]
    ## other allocations
    feature_vector = []
    edge_kernels   = []
    window         =(50,50)
    edge_kernels.append( np.array( [1,0,-1] ).reshape(1,3))
    edge_kernels.append( edge_kernels[0].T )
    edge_kernels.append( np.array( [ [0, 0.5, 1],  [-0.5, 0, 0.5], [-1,-0.5,0] ] ) )
    edge_kernels.append( np.flip(edge_kernels[-1], axis=0) )
    edge_kernels = [ kernel/ np.abs( kernel).sum() for kernel in edge_kernels]
    edge_kernels = [ fftn( embed_kernel( kernel, resolution)) for kernel in edge_kernels]
    #tic( 'rb loading and processing' )
    basis = DarusLoader().get_basis()
    for i in range( n_xi ):
        basis[:,i] = fftn( basis[:,i].reshape( resolution) ).flatten().real
    basis[0,:] = 0 #enforce 0 mean below in projection
    #toc( 'rb loading and processing' )

    #tic( f'full feature computation of {n_samples} samples', silent=False )
    #tic( 'fft', silent=True)
    fourier = np.zeros( (n_samples, *resolution), dtype=complex )
    for i in range( n_samples):
        fourier[i] = fftn( images[i] ) 
    #toc( 'fft')
    ### Volume fraction
    vol = images.mean(1).mean(1)
    feature_vector.append( vol[:,None])
    ###  the pcf is rescaled to 
    pcf_scaling = (vol * n_voxels**2 )[:,None,None]
    #tic( 'xi computation' , silent=True)
    feature_vector.append( ( (fourier*np.conj(fourier)/pcf_scaling ).reshape( n_samples, -1) @ basis).real )
    #toc( 'xi computation' )
    ### 
    #tic( 'band features' , silent=True)
    band_features = np.zeros( (n_samples, 2*n_angles ) )
    for j in range( n_angles):
      for i in range( n_samples):
        feature                 = ifftn( fourier[i] * diagonals[j]).real
        band_features[i, 2*j ]  = feature.max() 
        band_features[i, 2*j+1] = (1-feature.min()) 
    feature_vector.append( band_features)
    #toc( 'band features' )
    ###
    #tic( 'projected edges' , silent=True)
    feature_vector.append( projected_wall( images ) )
    #toc( 'projected edges' )
    ###
    #tic( 'local volume' , silent=True)
    feature_vector.append( vol_distribution( images ) )
    #toc( 'local volume' )
    ### 
    #tic( 'edge distributions' , silent=True)
    features = np.zeros( (n_samples, 3*len( edge_kernels) ) )
    for i in range( n_samples):
        for j in range( len( edge_kernels)):
            edges             = np.abs( ifftn( fourier[i] * edge_kernels[j]).real )
            feature_grid      = average_pooling( edges, window)
            features[i,3*j]   = np.mean( feature_grid) 
            features[i,3*j+1] = np.std( feature_grid) 
            features[i,3*j+2] = skew( feature_grid, axis=None) 
    feature_vector.append( features)
    ###
    #toc( 'edge distributions' )
    #toc( f'full feature computation of {n_samples} samples' )
    #print( 'length of each feature type:', [x.shape for x in feature_vector] )
    feature_vector = np.concatenate( feature_vector, axis=1)
    ### add the phase contrast as int feature if requested
    if isinstance( phase_contrasts, list) and len( phase_contrasts) > 1:
        multi_contrast = []
        for contrast in phase_contrasts:
            multi_contrast.append( np.insert( feature_vector, 1, contrast, axis=1)  )
        feature_vector = multi_contrast
    return feature_vector



### Dependend functions which are used for feature extraction but do not explicitely extract features
def average_pooling( image, kernel_size):
    """
    sum up local windows of a 2d-array (without overlap) and return it as 
    a smaller grid. Will always assert the same stride as kernel_size
    Parameters:
    -----------
    image:          numpy 2d-array
                    image data
    kernel_size:    int or tuple of ints
                    what window size it should consider in each step
    Returns:
    --------
    feature_map:    numpy 2d-array
                    feature map of size ceil( image.shape/ kernel_size )
    """
    size    = np.ceil( np.array(image.shape) / np.array( kernel_size) ).astype(int)
    feature = np.zeros( size)
    if isinstance( kernel_size, int):
        kernel_size = image.ndim*[ kernel_size] 
    for i in range( size[0] ):
        column = image[i*kernel_size[0]:(i+1)*kernel_size[0]]
        for j in range( size[1]):
            feature[i,j] = column[:,j*kernel_size[1]:(j+1)*kernel_size[1]].mean()
    return feature


def tilted_diagonal( n, width, angle, value=None):
    """
    Generate a quadratic matrix with a band structure through the image
    such that we have a line of width <width> through the center of the
    image. Does not take periodicity into account.
    Parameters:
    -----------
    n:          int
                size of matrix, will be of shape (n, n)
    width:      int
                size of the diagonal, resulting into 1+2*<width> band
    angle:      float
                angle (in rad) of the line through the center
    value:      float, default None
                value of the diagonal entries, defaults to 1/image.sum()
    """ 
    if angle > 2*pi:
        angle = angle*pi/180
    angle          = angle-pi/4 #make diagonal default
    n_dim          = 2
    image          = np.zeros( n_dim*[n] )
    origin         = np.array( n_dim*[0.5] ) 
    Q              = np.array( [ [np.cos( angle), -np.sin(angle)], [ np.sin(angle), np.cos( angle)] ] )
    a              = width /n 
    normal_vectors = np.array( [ [0, 1], [0, -1] ], dtype='float' )
    plane_points   = np.array( [ [ 0, a], [0,-a] ] )
    n_plane        = normal_vectors.shape[0]
    # Full rotation #requires for the "plane_points" to be also rotated
    normal_vectors  = normal_vectors @ Q
    plane_points    = plane_points @ Q
    shifted_center  = (plane_points + origin  ) *np.array( image.shape)
    xx, yy          = np.meshgrid( *[ range( x) for x in image.shape] )
    accepted_points = np.ones( image.shape, dtype=bool)
    for i in range( normal_vectors.shape[0] ):
        admissible_points = -(xx - shifted_center[i,0] )*normal_vectors[i,0] - (yy - shifted_center[i,1] )*normal_vectors[i,1]
        accepted_points   = np.logical_and( accepted_points, admissible_points >=0 )
    if value is None:
        image[ accepted_points] = 1
        return image/image.sum() 
    else:
        image[ accepted_points] = value
        return image


def embed_kernel( kernel, image_size=(400,400) ):
    """ embed the kernel required for convolution in fourier space """
    kernel                     = np.array( kernel) 
    kernel_shift               = np.array( [ floor( x/2) for x in kernel.shape ] )
    top_left                   = tuple( [ slice( x) for x in kernel.shape ] )
    embedded_kernel            = np.zeros( image_size)
    embedded_kernel[ top_left] = np.flip( kernel )
    embedded_kernel            = np.roll( embedded_kernel, -kernel_shift, axis=np.arange(kernel.ndim))
    return embedded_kernel

