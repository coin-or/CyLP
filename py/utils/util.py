import numpy as np
from math import atan2 
import Constants

def sign(x): 
    if x > 0 or (x == 0 and atan2(x, -1.) > 0.): 
        return 1 
    else: 
        return -1 

def get_cs(w1, w2):
    omega = float(sign(w1) * (w1**2 + w2**2)**0.5)
    c = w1 / omega
    s = w2 / omega
    return c, s

def givens(n, i, j, w1, w2):
    g = np.identity(n, float)
    if abs(w1) < Constants.EPSILON and abs(w2) < Constants.EPSILON:
        return g
    c, s = get_cs(w1, w2)
    g[i,i], g[j,j], g[i,j], g[j,i] = c, -c, s, s
    return g
    
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
      


                
