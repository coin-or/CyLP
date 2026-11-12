import warnings
import numpy as np
from functools import wraps
from math import atan2
from cylp.py import Constants
from operator import mul

def deprecatedWithQP(f):
    '''
    Mark *f* as deprecated. Only cylp.py.QP uses it, and it will be removed
    with cylp.py.QP.
    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        warnings.warn('cylp.py.utils.util.%s is deprecated and will be '
                      'removed in CyLP 0.95, along with cylp.py.QP.' %
                      f.__name__, DeprecationWarning, stacklevel=2)
        return f(*args, **kwargs)
    return wrapper

@deprecatedWithQP
def sign(x):
    if x > 0 or (x == 0 and atan2(x, -1.) > 0.):
        return 1
    else:
        return -1

@deprecatedWithQP
def get_cs(w1, w2):
    omega = float(sign(w1) * (w1**2 + w2**2)**0.5)
    c = w1 / omega
    s = w2 / omega
    return c, s

@deprecatedWithQP
def givens(n, i, j, w1, w2):
    g = np.identity(n, float)
    if abs(w1) < Constants.EPSILON and abs(w2) < Constants.EPSILON:
        return g
    c, s = get_cs(w1, w2)
    g[i,i], g[j,j], g[i,j], g[j,i] = c, -c, s, s
    return g

@deprecatedWithQP
def applyGivens(vec):
    'Applies dim-1 givens matrices so that vec contains only one non-zero element'
    v = vec.copy()
    dim = v.shape[0]
    Q_bar = np.matrix(np.identity(dim, float))
    Qlist = []
    for i in range(dim - 1):
        if vec[dim-i-1] != 0:
            Q = givens(dim, dim-i-2, dim-i-1, v[dim-i-2,0], v[dim-i-1,0])
            Qlist.append(Q)
            v = Q * v
            Q_bar = Q * Q_bar
    return Qlist, Q_bar, v[0,0]

@deprecatedWithQP
def UH2UT(mat):
    m = mat.copy()
    nrows, ncols = mat.shape
    Q_bar = np.matrix(np.identity(nrows, float))
    for i in range(ncols):
        Q = givens(nrows, i, i+1, m[i,i], m[i+1,i])
        m = Q * m
        Q_bar = Q * Q_bar
    #we need to remove the last line because it's all zero (and we need a square matrix)
    return Q_bar, m[:nrows-1, :]


## This part is for defining the decorators 'precondtion' and 'postcondition' and 'conditions'

__all__ = ['precondition', 'postcondition', 'conditions']

DEFAULT_ON = True

def precondition(precondition, use_conditions=DEFAULT_ON):
    return conditions(precondition, None, use_conditions)

def postcondition(postcondition, use_conditions=DEFAULT_ON):
    return conditions(None, postcondition, use_conditions)

class conditions(object):
    __slots__ = ('__precondition', '__postcondition')

    def __init__(self, pre, post, use_conditions=DEFAULT_ON):
        if not use_conditions:
            pre, post = None, None

        self.__precondition = pre
        self.__postcondition = post

    def __call__(self, function):
        # combine recursive wrappers (@precondition + @postcondition == @conditions)
        pres = set((self.__precondition,))
        posts = set((self.__postcondition,))

        # unwrap function, collect distinct pre-/post conditions
        while type(function) is FunctionWrapper:
            pres.add(function._pre)
            posts.add(function._post)
            function = function._func

        # filter out None conditions and build pairs of pre- and postconditions
        conditions = map(None, filter(None, pres), filter(None, posts))

        # add a wrapper for each pair (note that 'conditions' may be empty)
        for pre, post in conditions:
            function = FunctionWrapper(pre, post, function)

        return function

class FunctionWrapper(object):
    def __init__(self, precondition, postcondition, function):
        self._pre = precondition
        self._post = postcondition
        self._func = function

    def __call__(self, *args, **kwargs):
        precondition = self._pre
        postcondition = self._post

        if precondition:
            precondition(*args, **kwargs)
        result = self._func(*args, **kwargs)
        if postcondition:
            postcondition(result, *args, **kwargs)
        return result


class Ind:
    def __init__(self, key, dim):
        '''
        Create an instance of Ind using *key* that can be
        an integer, a slice, a list, or a numpy array. As in Python,
        negative indices count from the end.
        '''
        if isinstance(key, slice):
            self.indices = range(*key.indices(dim))
            if len(self.indices) == 0:
                raise IndexError('Indexing problem: %s, dim=%d:' % (str(key), dim))
        elif isinstance(key, (int, np.integer)):
            if not -dim <= key < dim:
                raise IndexError('Index (%d) out of range (%d)' % (key, dim))
            self.indices = [int(key) % dim]
        elif isinstance(key, (list, np.ndarray)):
            ind = np.asarray(key)
            if ind.size and ind.dtype.kind not in 'iu':
                raise TypeError('Error indexing with %s' % ind.dtype)
            outside = (ind < -dim) | (ind >= dim)
            if outside.any():
                raise IndexError('Index (%d) out of range (%d)' %
                                 (ind[outside][0], dim))
            self.indices = ind.astype(np.int64) % dim
        else:
            raise TypeError('Error indexing with unrecognized type %s' % key.__class__)
        self.dim = dim


    def __repr__(self):
        return '(%d, %d / %d)' % (self.start, self.stop, self.dim)

def getIndS(inds):
    n = len(inds)
    if n == 1:
        return inds[0].start
    prod = inds[0].stop
    for i in range(1, n):
        prod *= inds[i].dim
    return prod + getIndS(inds[1:])

def getMultiDimMatrixIndex(inds, res=[]):
    n = len(inds)
    r = inds[0].indices
    #r = range(inds[0].start, inds[0].stop)
    if n == 1:
        return r
    l = []
    for i in r:
        prod = i
        for k in range(1, n):
            prod *= inds[k].dim
        rest = getMultiDimMatrixIndex(inds[1:], res)
        l += res + [prod + rs for rs in rest]
    return l

def getTupleIndex(ind, dims):
    if isinstance(dims, int):
        return [ind] if ind < dims else -1
    n = len(dims)
    if ind > reduce(mul, dims):
        return -1
    if n == 1:
        return [ind]
    #return getTupleIndex(ind / dims[-1], dims[:-1]) + [ind % dims[-1]]
    ret = []
    for i in range(n):
        d = dims[n - i - 1]
        ret.insert(0, ind % d)
        ind /= d
        #return getTupleIndex(ind / dims[-1], dims[:-1]) + [ind % dims[-1]]
    return ret

if __name__ == '__main__':
    i1 = Ind(slice(1, 4), 5)
    i2 = Ind(slice(2, 4), 6)
    #i3 = Ind(slice(2, 5), 7)
    i3 = Ind(np.array([1, 4, 6]), 7)


    inds = getMultiDimMatrixIndex([i1, i2, i3])

    for i in inds:
        print(getTupleIndex(i, (5, 6, 7)), i)

    print(getTupleIndex(8, 8))
