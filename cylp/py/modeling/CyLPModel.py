'''
    Module provide a facility to model a linear program.
    Although the current usage is to pass the resulting LP to CLP
    to solve it is independant of the solver.

    .. _modeling-usage:

    **Usage (single dimension, using CyLPModel indirectly)**

    >>> import numpy as np
    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.modeling.CyLPModel import CyLPArray
    >>>
    >>> s = CyClpSimplex()
    >>>
    >>> # Add variables
    >>> x = s.addVariable('x', 3)
    >>> y = s.addVariable('y', 2)
    >>>
    >>> # Create coefficients and bounds
    >>> A = np.matrix([[1., 2., 0],[1., 0, 1.]])
    >>> B = np.matrix([[1., 0, 0], [0, 0, 1.]])
    >>> D = np.matrix([[1., 2.],[0, 1]])
    >>> a = CyLPArray([5, 2.5])
    >>> b = CyLPArray([4.2, 3])
    >>> x_u= CyLPArray([2., 3.5])
    >>>
    >>> # Add constraints
    >>> s += A * x <= a
    >>> s += 2 <= B * x + D * y <= b
    >>> s += y >= 0
    >>> s += 1.1 <= x[1:3] <= x_u
    >>>
    >>> # Set the objective function
    >>> c = CyLPArray([1., -2., 3.])
    >>> s.objective = c * x + 2 * y.sum()
    >>>
    >>> # Solve using primal Simplex
    >>> s.primal()
    'optimal'
    >>> s.primalVariableSolution['x']
    array([ 0.2,  2. ,  1.1])
    >>> s.primalVariableSolution['y']
    array([ 0. ,  0.9])
    >>> s += x[2] + y[1] >= 2.1
    >>> s.primal()
    'optimal'
    >>> s.primalVariableSolution['x']
    array([ 0. ,  2. ,  1.1])
    >>> s.primalVariableSolution['y']
    array([ 0.,  1.])

    **Usage (single dimension, using CyLPModel directly, depending on the
      size could be faster than the indirect approach)**

    >>> import numpy as np
    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
    >>>
    >>> model = CyLPModel()
    >>>
    >>> # Add variables
    >>> x = model.addVariable('x', 3)
    >>> y = model.addVariable('y', 2)
    >>>
    >>> # Create coefficients and bounds
    >>> A = np.matrix([[1., 2., 0],[1., 0, 1.]])
    >>> B = np.matrix([[1., 0, 0], [0, 0, 1.]])
    >>> D = np.matrix([[1., 2.],[0, 1]])
    >>> a = CyLPArray([5, 2.5])
    >>> b = CyLPArray([4.2, 3])
    >>> x_u= CyLPArray([2., 3.5])
    >>>
    >>> # Add constraints
    >>> model += A * x <= a
    >>> model += 2 <= B * x + D * y <= b
    >>> model += y >= 0
    >>> model += 1.1 <= x[1:3] <= x_u
    >>>
    >>> # Set the objective function
    >>> c = CyLPArray([1., -2., 3.])
    >>> model.objective = c * x + 2 * y.sum()
    >>>
    >>> # Create a CyClpSimplex instance from the model
    >>> s = CyClpSimplex(model)
    >>>
    >>> # Solve using primal Simplex
    >>> s.primal()
    'optimal'
    >>> s.primalVariableSolution['x']
    array([ 0.2,  2. ,  1.1])
    >>> s.primalVariableSolution['y']
    array([ 0. ,  0.9])
    >>> s += x[2] + y[1] >= 2.1
    >>> s.primal()
    'optimal'
    >>> s.primalVariableSolution['x']
    array([ 0. ,  2. ,  1.1])
    >>> s.primalVariableSolution['y']
    array([ 0.,  1.])

    **Usage (multi-dimensions, using CyLPModel indirectly)**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.modeling.CyLPModel import CyLPArray
    >>> s = CyClpSimplex()
    >>> x = s.addVariable('x', (5, 3, 6))
    >>> s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5
    >>> s += 0 <= x <= 1
    >>> c = CyLPArray(range(18))
    >>> s.objective = c * x[2, :, :] + c * x[0, :, :]
    >>> s.primal()
    'optimal'
    >>> sol = s.primalVariableSolution['x']
    >>> abs(sol[0, 1, 0] - 1) <= 10**-6
    True
    >>> abs(sol[2, 0, 3] - 1) <= 10**-6
    True

'''

from __future__ import print_function
from functools import reduce

# Python 3 does not have long, only int
try:
    long
except NameError:
    long = int

from itertools import product
from copy import deepcopy
from operator import mul
import numpy as np
from scipy import sparse
from scipy.sparse import identity, lil_matrix
from cylp.py.utils.util import Ind, getMultiDimMatrixIndex, getTupleIndex

NUMBERS = (int, float, long, np.int64, np.int32, np.double)
def isNumber(n):
    return (isinstance(n, NUMBERS) or
            (isinstance(n, CyLPArray) and n.shape == ()))

# Increase max stack size from 8MB to 512MB
#resource.setrlimit(resource.RLIMIT_STACK, (2**5,-1))
#sys.setrecursionlimit(10**6)

#from cylp.py.utils.sparseUtil import sparseConcat
def I(n):
    '''
    Return a sparse identity matrix of size *n*
    '''
    if n <= 0:
        return None
    #return csc_matrixPlus(sparse.eye(n, n))
    return identity(n).tocsr()

def identitySub(var):
    '''
    Return a sparse row sub-matrix of the identity matrix
    for ``var``.
    '''
    n = var.dim
    if var.parent is not None:
        n = var.parent.dim
    if var.dim == 1:
        m = lil_matrix((1, n))
        m[0, var.indices[0]] = 1
        return m
    return I(n)[var.indices, :]

class CyLPExpr:
    operators = ('>=', '<=', '==', '+', '-', '*', 'u-', 'sum')

    def __init__(self, opr='', left='', right=''):
        self.opr = opr
        self.left = left
        self.right = right
        self.expr = self
        #self.variables = []

    def __hash__(self):
        return id(self)

    def __repr__(self):
        s = '%s %s %s' % (self.left, self.right, self.opr)
        return s

    def __le__(self, other):
        v = CyLPExpr(opr="<=", right=other, left=self.expr)
        self.expr = v
        return v

    def __ge__(self, other):
        v = CyLPExpr(opr=">=", right=other, left=self.expr)
        self.expr = v
        return v

    def __eq__(self, other):
        # Check if both sides are CyLPVar, in which case a pythonic
        # comparison is meant (for dictionary keys,...)
        if (other is None):
            return False
        if isinstance(self, CyLPVar) and isinstance(other, CyLPVar):
            return id(self) == id(other)
            #return (str(self) == str(other))
        v = CyLPExpr(opr="==", right=other, left=self.expr)
        self.expr = v
        return v

    def __rmul__(self, other):
        v = CyLPExpr(opr="*", left=other, right=self)
        v.expr = v
        self.expr = v
        return v

    def __mul__(self, other):
        v = CyLPExpr(opr="*", left=self, right=other)
        v.expr = v
        self.expr = v
        return v

    def __rsub__(self, other):
        v = CyLPExpr(opr="-", left=other, right=self)
        v.expr = v
        self.expr = v
        return v

    def __sub__(self, other):
        v = CyLPExpr(opr="-", left=self, right=other)
        v.expr = v
        self.expr = v
        return v

    def __radd__(self, other):
        v = CyLPExpr(opr="+", left=other, right=self)
        v.expr = v
        self.expr = v
        return v

    def __add__(self, other):
        v = CyLPExpr(opr="+", left=self, right=other)
        v.expr = v
        self.expr = v
        return v

    def __neg__(self):
        v = CyLPExpr(opr="u-", right=self)
        v.expr = v
        self.expr = v
        return v

    def getPostfix(self):
        if isinstance(self, CyLPVar):
            return [self]
        left = [self.left]
        right = [self.right]
        if isinstance(self.left, CyLPExpr):
            left = self.left.getPostfix()
        if isinstance(self.right, CyLPExpr):
            right = self.right.getPostfix()

        return left + right + [self.opr]

    def evaluate(self, name=''):
        '''
        Evaluates an expression in the postfix form
        '''
        #FIXME: sometimes calling this twice is different than running it once
        tokens = self.getPostfix()
        operands = []

        cons = CyLPConstraint(name)

        for token in tokens:
            # If an operand found
            if type(token) != str or token not in CyLPExpr.operators:
                operands.append(token)
                if (isinstance(token, CyLPVar)):
                    varToBeAdded = token
                    if not varToBeAdded in cons.variables:
                        cons.variables.append(varToBeAdded)

            # Else we know what to do with an operator
            else:
                if token == 'u-':
                    right = operands.pop()
                    cons.perform(token, right=right)
                    operands.append(CyLPExpr(token, right=right))
                else:
                    right = operands.pop()
                    left = operands.pop()
                    cons.perform(token, left, right)
                    operands.append(CyLPExpr(token, left, right))

        # If something remains in the *operands* list, it means
        # that we have a single CyLPVar object on a line with no
        # operators operating on it. Check this situation and
        # add a 1 vector as its coefficient
        if len(operands) == 1 and isinstance(operands[0], CyLPVar):
            var = operands[0]
            cons.varCoefs[var] = identitySub(var) #np.ones(var.dim)

        # Reset variables for future constraints
        for var in cons.varCoefs.keys():
            var.expr = var

        return cons


class CyLPConstraint:
    gid = 0
    def __init__(self, name=''):
        self.varNames = []
        self.parentVarDims = {}
        self.varCoefs = {}
        self.lower = None
        self.upper = None
        self.nRows = None
        self.isRange = True
        self.variables = []
        if not name:
            CyLPConstraint.gid += 1
            name = 'R_%d' % CyLPConstraint.gid
        self.name = name

    def __repr__(self):
        s = '\n'
        s += 'constraint %s:\n' % self.name
        s += 'variable names:\n'
        s += str(self.varNames) + '\n'
        s += 'coefficients:\n'
        s += str(self.varCoefs) + '\n'
        s += 'lower = %s\n' % str(self.lower)
        s += 'upper = %s\n' % str(self.upper)
        if self.isRange:
            s += 'Constraint is a range\n'
        else:
            s += 'normal Constarint\n'
        return s

    def mul(self, expr, coef):
        '''
        Recursively multiplies all variable coefficients in *expr* by *coef*
        '''
        if isinstance(expr, CyLPVar):
            self.varCoefs[expr] *= coef
            return
        if isinstance(expr, CyLPExpr):
            if expr.left is None:
                return
            self.mul(expr.right, coef)
            self.mul(expr.left, coef)


    def perform(self, opr, left=None, right=None):
        if isinstance(left, (CyLPArray, np.matrix, np.ndarray)):
            left = CyLPArray(np.squeeze(np.asarray(left)))
        if isinstance(right, (CyLPArray, np.matrix, np.ndarray)):
            right = CyLPArray(np.squeeze(np.asarray(right)))

        if isinstance(right, CyLPVar):
            if right.dim == 0:
                return
            if right.name not in self.varNames:
                self.varNames.append(right.name)
                self.parentVarDims[right.name] = right.parentDim
            if opr == 'u-':
                #ones = -CyLPArray(np.ones(right.dim))
                #self.nRows = right.dim
                #self.varCoefs[right] = ones
                self.varCoefs[right] = -identitySub(right)
                self.nRows = len(right.indices)
                self.isRange = False
            elif opr == 'sum':
                n = right.dim
                if right.parent is not None:
                    n = right.parent.dim
                self.varCoefs[right] = sparse.lil_matrix((1, n))
                self.varCoefs[right][0, right.indices] = np.ones(len(right.indices))
                self.nRows = 1
                self.isRange = False
            else:
                #self.varNames.append(right.name)
                #self.parentVarDims[right.name] = right.parentDim
                dim = right.dim

                if opr == "*":  # Setting the coefficients
                    # Ignore a term if its coef is None
                    if isNumber(left):
                        if self.nRows and self.nRows != 1:
                            raise Exception("Expected 1-dimensional" \
                                            " coefficient")
                        #self.nRows = 1
                        #coef = CyLPArray(left * np.ones(dim))
                        coef = left * identitySub(right)
                        self.nRows = len(right.indices)
                    else:
                        if left is None:
                            return
                        if right.dim != left.shape[-1]:
                            raise Exception("Coefficient:\n%s\n has %d" \
                                            " columns expected %d"
                                        % (left, left.shape[1], right.dim))
                        if len(left.shape) == 1:  # Array
                            nr = 1
                        else:
                            nr = left.shape[0]
                            #nr = len(left)
                        if self.nRows and self.nRows != nr:
                            raise Exception("Coefficient:\n%s\n has %d rows" \
                                            " expected %d"
                                        % (left, left.shape[0], self.nRows))
                        self.nRows = nr
                        if right.parent is None:
                            coef = deepcopy(left)
                        else:
                            coef = sparse.lil_matrix((nr, right.parent.dim))
                            coef[:, right.indices] = left

                    self.varCoefs[right] = coef
                    self.isRange = False

                if opr == '+' or opr == '-':
                    # No coefs for left op
                    self.isRange = False
                    if isinstance(left, CyLPVar):
                        self.varCoefs[left] = identitySub(left)
                        self.nRows = len(left.indices)
                        if left.name not in self.varNames:
                            self.varNames.append(left.name)
                            self.parentVarDims[left.name] = left.parentDim

                    # No coefs for right op
                    if isinstance(right, CyLPVar):
                        if right in self.varCoefs.keys():
                            self.varCoefs[right] *= -1
                        else:
                            coef = identitySub(right)
                            if opr == '-':
                                coef *= -1
                            self.varCoefs[right] = coef
                            self.nRows = len(right.indices)
                            #if self.nRows == 0:
                            #    self.nRows = 1
                            #    coef = CyLPArray(np.ones(right.dim))
                            #elif self.nRows == 1:
                            #    coef = CyLPArray(np.ones(1))
                            #else:
                            #    coef = np.matrix(np.eye(self.nRows))

                            #if opr == '-':
                            #    coef *= -1
                            #self.varCoefs[right] = coef
                            if right.name not in self.varNames:
                                self.varNames.append(right.name)
                                self.parentVarDims[right.name] = \
                                                            right.parentDim
        # check if no coef for left
        if left.__class__ == CyLPVar and (opr == '-' or opr == '+'):
            if left.dim == 0 :
                return
            self.isRange = False
            #ones = CyLPArray(np.ones(left.dim))
            #self.varCoefs[left] = ones
            #self.nRows = 1
            self.varCoefs[left] = identitySub(left)
            self.nRows = len(left.indices)
            if left.name not in self.varNames:
                self.varNames.append(left.name)
                self.parentVarDims[left.name] = left.parentDim

        # Coef already set for the right operand
        if right.__class__ == CyLPExpr:
            if opr == '-':
                # The expression on the right is in the form (left opr right)
                # so the key to the coef in self.varCoefs is right.right i.e.
                # the CyLPVar on the right
                #if right.right.dim == 0:
                #    return
                self.isRange = False
                self.mul(right, -1)
                #self.varCoefs[right.right] *= -1
            if opr == '*' and isNumber(left):
                self.mul(right, left)


        if opr in ('<=', '>=', '=='):
            if isinstance(left, CyLPExpr) and not isinstance(right,
                                                              CyLPExpr):
                bound = right
            elif isinstance(right, CyLPExpr) and not isinstance(left,
                                                                CyLPExpr):
                bound = left
            else:
                raise Exception('At least one side of a comparison sign' \
                                'should be constant.')

            # FIXME: check this: I suppose never runs
            if self.isRange:
                var = self.variables[0]
                self.nRows = var.dim
                dim = var.dim
            #if not self.nRows:
                #self.nRows = left.parentDim
                #dim = left.parentDim
            else:
#                # FIXME: Think of the correct way to check bound dimensions
#                if (not isinstance(right, (float, long, int)) and
#                            right.shape[0] != self.nRows):
#                    #raise Exception('Bound dim: %d, expected dim: %d '
#                    #                % (right.shape[0], self.nRows))
#                    pass
                dim = self.nRows
#
            if isNumber(right):
                bound = CyLPArray(right * np.ones(dim))

            if self.isRange:
                if ((opr in ('>=', '==') and isinstance(left, CyLPExpr)) or
                    (opr in ('<=', '==') and isinstance(right, CyLPExpr))):
                    if var.parent:
                        var.parent.lower[var.indices] = bound
                    else:
                        var.lower[var.indices] = bound

                if ((opr in ('<=', '==') and isinstance(left, CyLPExpr)) or
                    (opr in ('>=', '==') and isinstance(right, CyLPExpr))):
                    if var.parent:
                        var.parent.upper[var.indices] = bound
                    else:
                        var.upper[var.indices] = bound

            else:
                if ((opr in ('>=', '==') and isinstance(left, CyLPExpr)) or
                    (opr in ('<=', '==') and isinstance(right, CyLPExpr))):
                    self.lower = bound
                    if self.upper is None:
                        self.upper = getCoinInfinity() * np.ones(len(bound))
                if ((opr in ('<=', '==') and isinstance(left, CyLPExpr)) or
                    (opr in ('>=', '==') and isinstance(right, CyLPExpr))):
                    self.upper = bound
                    if self.lower is None:
                        self.lower = -getCoinInfinity() * np.ones(len(bound))


class CyLPVar(CyLPExpr):
    '''
    Contains variable information such as its name, dimension,
    bounds, and whether or not it is an integer. We don't creat
    instances directly but rather use :py:func:`CyLPModel.addVariable`
    to create one.

    See the :ref:`modeling example <modeling-usage>`.

    '''

    def __init__(self, name, dim, isInt=False, fromInd=None, toInd=None):
        self.name = name
        self.dims = None
        if isinstance(dim, tuple):
            self.setDims(dim)
            self.dim = dim = reduce(mul, dim)
            self.parentDim = dim
        else:
            self.dim = dim
            self.parentDim = dim
        self.parent = None
        self.isInt = isInt
        self.expr = self
        self.lower = -getCoinInfinity() * np.ones(dim)
        self.upper = getCoinInfinity() * np.ones(dim)

        self.indices = np.arange(dim, dtype=np.int32)
        self.fromInd = self.toInd = None
        if fromInd and toInd:
            self.indices = np.arange(fromInd, toInd, dtype=np.int32)
            self.formInd = fromInd
            self.toInd = toInd

    def __repr__(self):
        s = self.name
        if self.fromInd and self.toInd:
            s += '[%d:%d]' % (self.fromInd, self.toInd)
        elif self.parent is not None and len(self.indices) == 1:
            s += '[%d]' % self.indices
        return s

    def setDims(self, ds):
        self.dims = ds

    def __getitem__(self, key):
        if type(key) == int:
            newObj = CyLPVar(self.name, 1)
            newObj.indices = np.array([key], np.int32)
            newObj.parentDim = self.dim
            newObj.parent = self
            newObj.dim = 1
        elif type(key) == slice:
            newObj_fromInd = key.start
            if key.stop > self.dim:
                newObj_toInd = self.dim
            else:
                newObj_toInd = key.stop

            newObj = CyLPVar(self.name, newObj_toInd - newObj_fromInd)
            newObj.parentDim = self.dim
            newObj.parent = self
            newObj.fromInd = newObj_fromInd
            newObj.toInd = newObj_toInd
            newObj.indices = np.arange(newObj.fromInd, newObj.toInd,
                                       dtype='int32')
            # Save parentDim for future use
            newObj.dim = len(newObj.indices)
        elif isinstance(key, (np.ndarray, list)):
            newObj = CyLPVar(self.name, len(key))
            newObj.parentDim = self.dim
            newObj.parent = self
            newObj.fromInd = None
            newObj.toInd = None
            newObj.indices = np.array(key)
            newObj.dim = len(newObj.indices)

        elif isinstance(key, tuple):
            inds = []
            n = range(len(key))
            for i in n:
                k = key[i]
                if isinstance(k, int):
                    inds.append(Ind(slice(k, k+1), self.dims[i]))
                else:
                    inds.append(Ind(k, self.dims[i]))
            newObj = self.__getitem__(getMultiDimMatrixIndex(inds))
        return newObj

    def sum(self):
        v = CyLPExpr(opr="sum", right=self)
        v.expr = v
        self.expr = v
        return v


class CyLPArray(np.ndarray):
    '''
    It is a tweaked Numpy array. It allows user to define objects
    comparability to an array. If we have an instance of ``CyLPVar``
    called ``x`` and a numpy array ``b``, then ``b >= x`` returns an array
    of the same dimension as ``b`` with all elements equal to ``False``,
    which has no sense at all. On the constrary ``CyLPArray`` will return
    ``NotImplemented`` if it doesn't know how to compare. This is particularly
    helpful when we define LP constraints.

    See the :ref:`modeling example <modeling-usage>`.

    '''

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.__le__ = CyLPArray.__le__
        obj.__ge__ = CyLPArray.__ge__
        return obj

    def __le__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__le__(self, other)

    def __ge__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__ge__(self, other)

    def __mul__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__mul__(self, other)

    def __rmul__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__rmul__(self, other)

    def __add__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__add__(self, other)

    def __radd__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__radd__(self, other)

    def __rsub__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__rsub__(self, other)

    def __sub__(self, other):
        if (issubclass(other.__class__, CyLPExpr)):
            return NotImplemented
        return np.ndarray.__sub__(self, other)


class IndexFactory:
    '''
    A small class that keeps track of the indices of variables and constraints
    that you are adding to a problem gradually.

    **Usage**

    >>> import numpy as np
    >>> from cylp.py.modeling.CyLPModel import IndexFactory
    >>> i = IndexFactory()
    >>> i.addVar('x', 10)
    >>> i.addVar('y', 5)
    >>> i.hasVar('x')
    True
    >>> (i.varIndex['y'] == np.arange(10, 15)).all()
    True
    >>> i.addConst('Manpower', 4)
    >>> i.addConst('Gold', 10)
    >>> (i.constIndex['Manpower'] == np.arange(4)).all()
    True
    >>> (i.constIndex['Gold'] == np.arange(4, 14)).all()
    True

    '''

    def __init__(self):
        self.varIndex = {}
        self.constIndex = {}
        self.currentVarIndex = 0
        self.currentConstIndex = 0

    def addVar(self, varName, numberOfVars):
        if numberOfVars == 0:
            return
        if not varName:
            raise Exception('You must specify a name for a variable.')
        if varName in self.varIndex.keys():
            print('Variable already exists.')
            #self.varIndex[varName] += range(self.currentVarIndex,
            #                                self.currentVarIndex +
            #                                numberOfVars)
        else:
            self.varIndex[varName] = np.arange(self.currentVarIndex,
                                            self.currentVarIndex +
                                            numberOfVars, dtype='int32')
        self.currentVarIndex += numberOfVars

    def removeVar(self, name):
        if not self.hasVar(name):
            raise Exception('Variable "%s" does not exist.' % name)
        nVars = len(self.varIndex[name])
        start = self.varIndex[name][0]
        del self.varIndex[name]
        for varName in self.varIndex.keys():
            inds = self.varIndex[varName]
            if inds[0] > start:
                self.varIndex[varName] = inds - nVars * np.ones(len(inds),
                                                                np.int32)

    def hasVar(self, varName):
        return varName in self.varIndex.keys()

    def hasConst(self, constName):
        return constName in self.constIndex

    def getLastVarIndex(self):
        return self.currentVarIndex - 1

    def addConst(self, constName, numberOfConsts):
        if numberOfConsts == 0:
            return
        if not constName:
            raise Exception('You must specify a name for a constraint.')
        if self.hasConst(constName):
            print('Constraint already exists: %s' % constName)
            #self.constIndex[constName] += range(self.currentConstIndex,
            #        self.currentConstIndex + numberOfConsts)
        else:
            self.constIndex[constName] = np.arange(self.currentConstIndex,
                                            self.currentConstIndex +
                                            numberOfConsts, dtype='int32')
        self.currentConstIndex += numberOfConsts

    def removeConst(self, name):
        if not self.hasConst(name):
            raise Exception('Constraint "%s" does not exist.' % name)
        nCons = len(self.constIndex[name])
        start = self.constIndex[name][0]
        del self.constIndex[name]
        for constName in self.constIndex.keys():
            inds = self.constIndex[constName]
            if inds[0] > start:
                self.constIndex[constName] = inds - nCons * np.ones(len(inds),
                                                                    np.int32)
        self.currentConstIndex -= nCons


    def getLastConstIndex(self):
        return self.currentConstIndex - 1

    def __repr__(self):
        varind = self.varIndex
        cind = self.constIndex

        s = "variables : \n"
        for vname, rg in varind.items():
            s += '%s : %s\n' % (vname.rjust(15), str(rg))
        s += '\n'
        s += "constraints : \n"
        for cname, rg in cind.items():
            s += '%s : %s\n' % (cname.rjust(15), str(rg))
        return s

    def reverseVarSearch(self, ind):
        '''
        Take an index and return the corresponding variable name.
        '''
        inds = self.varIndex
        for key in inds.keys():
            if ind in inds[key]:
                i = np.where(inds[key]==ind)[0][0]
                return key, i
        return -1, -1


from cylp.py.utils.sparseUtil import sparseConcat, csr_matrixPlus, csc_matrixPlus

class CyLPModel(object):
    '''
    Hold all the necessary information to create a linear program, i.e.
    variables, constraints, and the objective function.

    See the :ref:`modeling example <modeling-usage>`.

    '''

    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective_ = None
        self.inds = IndexFactory()
        self.nVars = 0
        self.varNames = []
        self.nCons = 0
        self.pvdims = {}

    def addVariable(self, name, dim, isInt=False):
        '''
        Create a new instance of :py:class:`CyLPVar` using the given
        arguments and add it to current model's variable list.
        '''

        if dim == 0:
            return
        var = CyLPVar(name, dim, isInt)
        self.variables.append(var)

        #If mulidim, correct dim
        if isinstance(dim, tuple):
            dim = reduce(mul, dim)

        if not self.inds.hasVar(var.name):
            self.inds.addVar(var.name, dim)
            self.nVars += dim
            self.varNames.append(var.name)
            self.pvdims[var.name] = dim

            if var.dims:
                var.mpsNames = [var.name + '_' + '_'.join(x) for x in \
                        product(*[map(str, range(i)) for i in var.dims])]
            else:
                var.mpsNames = ['%s_%s' % (var.name, i) for i in range(var.dim)]

            o = self.objective_
            if isinstance(o, np.ndarray):
                o = np.concatenate((o, np.zeros(dim)), axis=0)
            else:
                o = sparseConcat(o, csr_matrixPlus(np.zeros(dim)), 'h')

            # I'm not exactly sure why the objective gets changed into
            # csr_matrixPlus here when it may not be coming in. Shouldn't it
            # just always be?
            self.objective_ = csr_matrixPlus(o)

        else:
            raise Exception('Varaible %s already exists.' % var.name)

        return var

    def removeVariable(self, name):
        '''
        Remove a variable named ``name`` from the model
        '''
        if not self.inds.hasVar(name):
            raise Exception('Variable "%s" does not exist.' % name)

        self.nVars -= self.pvdims[name]
        start = self.inds.varIndex[name][0]
        end = start + self.pvdims[name]
        o = self.objective_

        if isinstance(o, np.ndarray):
            o = np.concatenate((o[:start], o[end:]), axis=0)
        else:
            if end == o.shape[1]:
                if start == 0:
                    print('Problem empty.')
                else:
                    o = o[0, :start]
            elif start == 0:
                o = o[0, end:]
            else:
                o = sparseConcat(o[0, :start], o[0, end:], how='h')

        # I'm not exactly sure why the objective gets changed into
        # csr_matrixPlus here when it may not be coming in. Shouldn't it
        # just always be?
        self.objective_ = csr_matrixPlus(o)

        del self.pvdims[name]
        self.varNames.remove(name)
        self.inds.removeVar(name)
        for i in range(len(self.variables)):
            var = self.variables[i]
            if var.name == name:
                del self.variables[i]
                break

        #Removing the variable from the constraints
        cons = self.constraints
        for c in cons:
            if name in c.varNames:
                c.varNames.remove(name)
                del c.parentVarDims[name]
                for v in list(c.varCoefs.keys()):
                    if v.name == name:
                        del c.varCoefs[v]

        for c in cons[:]:
            self.inds
            if not c.varCoefs:
                self.removeConstraint(c.name)

    def getVarByName(self, varName):
        '''
        Return a variable with name ``varName``.
        '''
        for var in self.variables:
            if var.name == varName:
                return var
        return None

    def makeIndexFactory(self):
        self.allVarNames = []
        varNameDict = {} #copy of allVarNames in dictionary form, for efficiency
        self.allParentVarDims = {}
        self.nRows = 0
        for c in self.constraints:
            # Make sure to add variables in the same order they were
            # added to CyLPModel
            for var in self.variables:
                for vname in c.varNames:
                    if vname == var.name and vname not in varNameDict:
                        self.allVarNames += [vname]
                        varNameDict[vname] = True
            self.allParentVarDims.update(c.parentVarDims)
            self.nRows += c.nRows

    @property
    def objective(self):
        return self.objective_

    @objective.setter
    def objective(self, obj):
        if isinstance(obj, CyLPExpr):
            self.objective_ = obj.evaluate()
            obj = None
            #for varName in self.allVarNames:
            for varName in self.varNames:
                v_coef = self.generateVarObjCoef(varName)
                obj = sparseConcat(obj, v_coef, how='h')
            obj = np.squeeze(np.asarray(obj.todense()))
        self.objective_ = obj

    def evaluateAt(self, x0):
        return np.dot(self.objective, x0)

    def __iadd__(self, cons):
        '''
        Call :meth:`addConstraint`. The only difference is that
        you cannot specify a name for your constraint.
        '''
        self.addConstraint(cons)
        return self

    def addConstraint(self, cons, consName='', addMpsNames=True):
        '''
        Add constraint ``cons`` to the ``CyLPModel``. Argument ``cons`` must
        be an expresion made using instances of :py:class:`CyLPVar`,
        :py:class:`CyLPArray`, Numpy matrices, or sparse matrices. It can
        use normal operators such as ``>=``, ``<=``, ``==``, ``+``, ``-``,
        ``*``.

        See the :ref:`modeling example <modeling-usage>`.

        '''

        c = cons.evaluate(consName)
        if not c.isRange:
            self.constraints.append(c)
            if c.name:
                self.inds.addConst(c.name, c.nRows)
            self.nCons += c.nRows

        if addMpsNames:
            c.mpsNames = []
            for i in range(c.nRows):
                c.mpsNames.append('%s_%s' % (c.name, str(i)))


        #self.makeMatrices()
        return c

    def removeConstraint(self, name):
        '''
        Remove constraint called ``name`` and return its CLP indices back
        for actual removal.
        '''
        if not self.inds.hasConst(name):
            raise Exception('Constraint "%s" does not exist.' % name)

        for i in range(len(self.constraints)):
            if self.constraints[i].name == name:
                con = self.constraints[i]
                break
        self.nCons -= con.nRows
        del self.constraints[i]
        indsOfConstriantsToBeRemoved = self.inds.constIndex[name]
        self.inds.removeConst(name)
        return indsOfConstriantsToBeRemoved

    def generateVarObjCoef(self, varName):
        #dim = self.allParentVarDims[varName]
        dim = self.pvdims[varName]
        coef = csr_matrixPlus((1, dim))
        obj = self.objective_
        keys = [k for k in obj.varCoefs.keys() if k.name == varName]

        for var in keys:
            coef = coef + obj.varCoefs[var]
        return coef

    def generateVarMatrix(self, varName):
        '''
        Create the coefficient matrix for a variable named
        varName. Return a csr_matrixPlus.
        '''
        #dim = self.allParentVarDims[varName]
        dim = self.pvdims[varName]
        mainCoef = None
        for c in self.constraints:
            coef = sparse.coo_matrix((c.nRows, dim))
            keys = [k for k in c.varCoefs.keys() if k.name == varName]
            for var in keys:
                coef = coef + c.varCoefs[var]
            mainCoef = sparseConcat(mainCoef, coef, 'v')

        return mainCoef

    def makeMatrices(self):
        '''
        Makes coef matrix and rhs vector from CyLPConstraints
        in self.constraints
        '''
        #if len(self.constraints) == 0:
        #    return
        self.makeIndexFactory()
        #self.getVarBounds()


        # Create the aggregated coef matrix
        masterCoefMat = None

        masterCoefMat = None
        if self.nCons > 0:
            for varName in self.varNames:# self.pvdims.keys():#self.allVarNames:
                vmat = self.generateVarMatrix(varName)

                if vmat is None:
                    vmat = csc_matrixPlus((self.nCons, self.pvdims[varName]))
                masterCoefMat = sparseConcat(masterCoefMat, vmat, 'h')

        # Create bound vectors
        c_lower = np.array([])
        c_upper = np.array([])
        for c in self.constraints:
            if not c.isRange:
                c_lower = np.concatenate((c_lower, c.lower), axis=0)
                c_upper = np.concatenate((c_upper, c.upper), axis=0)
        # Create variables bound vectors
        v_lower = np.array([])
        v_upper = np.array([])
        if self.nVars:
            v_lower = -getCoinInfinity() * np.ones(self.nVars)
            v_upper = getCoinInfinity() * np.ones(self.nVars)
            for v in self.variables:
                varinds = self.inds.varIndex[v.name]
                v_lower[varinds] = v.lower
                v_upper[varinds] = v.upper

                #v_lower = np.concatenate((v_lower, v.lower), axis=0)
                #v_upper = np.concatenate((v_upper, v.upper), axis=0)
        #if masterCoefMat is not None:
        return masterCoefMat, c_lower, c_upper, v_lower, v_upper


class CyLPSolution:

    def __init__(self):
        self.sol = {}

    def add(self, key, val):
        'Add (key, val) to self.dic. Ignore if val is zero'
        if val == 0: # See if necessary to use a tolerance
            return
        self.sol[key] = val

    def getVal(self, key):
        'Return the value corresponing key'
        if key not in self.sol.keys():
            return 0
        return self.sol[key]

    def __getitem__(self, key):
        #if isinstance(key, tuple):
        if key in self.sol.keys():
            return self.sol[key]
        else:
            return 0

    def __setitem__(self, key, val):
        #if isinstance(key, tuple):
        if val == 0: # See if necessary to use a tolerance
            return
        self.sol[key] = val
    def __repr__(self):
        return repr(self.sol)

def getCoinInfinity():
    return 1.79769313486e+308


if __name__ == '__main__':
    from cylp.cy import CyClpSimplex
    from cylp.py.modeling.CyLPModel import CyLPArray
    s = CyClpSimplex()
    x = s.addVariable('x', (5, 3, 6))
    s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5

    s += 0 <= x <= 1
    c = CyLPArray(range(18))

    s.objective = c * x[2, :, :] + c * x[0, :, :]
    s.writeMps('/Users/mehdi/Desktop/test.mps')
    s.primal()
    sol = s.primalVariableSolution
    print(sol)

#model = CyLPModel()
#
#x = model.addVariable('x', 5)
#y = model.addVariable('y', 4)
#z = model.addVariable('z', 5)
#
#
#b = CyLPArray([3.1, 4.2])
#aa = np.matrix([[1, 2, 3, 3, 4], [3, 2, 1, 2, 5]])
#aa = np.matrix([[0, 0, 0, -1.5, -1.5], [-3, 0, 0, -2,0 ]])
#dd = np.matrix([[1, 3, 1, 4], [2, 1,3, 5]])
#
#
#model.addConstraint(1.5 <=  x[1:3] + 5 * y[3:] - 6 * x[2:5]  + x[0] <= 14)
#
#print(model.constraints[0].varCoefs)
#
#model.addConstraint(2 <= aa*x + dd * y <= b)
#model.addConstraint(x[2:4] >= 3)
#model.addConstraint(1.1 <= y <= CyLPArray([1,2,3,4]))
#
#
#A = model.makeMatrices()
#
#print('x:')
#print(x.lower)
#print(x.upper)
#print('y:')
#print(y.lower)
#print(y.upper)
#
#print('+++++++++++++++++++++++++++++')
##cc = x[0] - 2*x[1:3] -2.5 * y[1]
##cc = cc.evaluate()
#model.objective = x[0] - 2*x[1:3] -2.5 * y[1]
##print(model.allParentVarDims)
##print(model.generateVarObjCoef('x'))
##print(model.generateVarObjCoef('y'))
#
#print(model.objective)
##print(cl)
##print(cu)
