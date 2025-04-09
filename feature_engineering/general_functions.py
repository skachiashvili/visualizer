import numpy as np
import time

class Cycler():
    """
    define a cycling iterator of an iterator which returns 2 values
    upon next() invokation. Function is intended to be used for
    torch.utils.data.DataLoader for an endless cycling
    This class is only meant to be used with <next( Cycler)>
    """
    def __init__( self, iterator, iteration_type=iter):
        """
        Allocate the iterating object as well as a copy of the <iterator>
        Doubles the memory requirement for the <iterator> object!
        Parameters:
        -----------
        iterator:           python object
                            python object to iterate with
        iteration_type:     function, default iter
                            type of iterator to store, governs the return of
                            next(). Use e.g. iter or enumerate
        """
        self.original_iterator = iterator
        self.iterator = iteration_type( iterator)
        self.iteration_type = iteration_type

    def __next__( self):
        try:
            return next( self.iterator)
        except:
            self.iterator = self.iteration_type( self.original_iterator)
            return next( self.iterator)

    def __getitem__(self, idx):
        return self.original_iterator[idx]

    def __iter__(self):
        return iter( self.original_iterator )

    def __len__(self):
        return len( self.original_iterator)


## timing related functions
initialized_times = dict() 
def tic( tag='', silent=False):
    """
    initializes the tic timer
    different tags allow for tracking of different computations
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    silent:     bool, default False
                Whether or not initialization should be printed
    """
    initialized_times[tag] = time.time()
    if not silent:
        print( 'Initializing timer for this tag:', tag)

def toc( tag='', precision=4 ):
    """
    prints the time passed since the invocation of the tic tag 
    does not remove the tag on call, can be timed multiple times 
    since start
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    precision:  int, default 4
                How many digits after ',' are printed
    """
    time_passed = time.time() - initialized_times[tag]
    try:
        print( '{1} -> elapsed time:{2: 0.{0}f}'.format( precision, tag, time_passed) )
    except:
        print( 'tic( tag) not specified, command will be ignored!')



### other generally usable functions
def bin_indices( x, x_binned=None, n_bins=15 ):
    """
    Find the indices in <x> which are inside each bin defined in x_binned.
    If <x_binned> is not specified, <x> is binned into <n_bins> uniform
    bins.  
    This function might not work for descending x_binned, has not been tested.
    Parameters:
    -----------
    x:              numpy 1d-array
                    given data
    x_binned:       numpy 1d-array
                    bin bounds to which the data should be sorted
    n_bins:         int, default 15
                    How many bins the data should be put into
                    if x_binned is not specified 
    Returns:
    --------
    bin_indices:    list of numpy 1d-arrays
                    indices for each bin, has of length len( x_binned)
    x_binned:       OPTIONAL, numpy 1d-array
                    bin intervals if x_binned was not given as input 
    """
    if x_binned is None:
        given_bins = False
        bin_incr = (x.max() - x.min() )/ (n_bins-1)
        x_binned = np.arange( x.min(), x.max() + 0.5*bin_incr, bin_incr)
        assert len( x_binned) == n_bins, 'got more bins than requested, got {}, wanted {}'.format( len( x_binned), n_bins) 
    else:
        given_bins = True
        n_bins = len( x_binned)
    x_idx = []
    for i in range( n_bins-1):
        x_idx.append( np.argwhere( (x_binned[i] < x) * (x <= x_binned[i+1]) ) )
    x_idx.append( np.argwhere( x_binned[-2] < x ) )
    if given_bins:
        return x_idx
    else:
        return x_idx, x_binned
