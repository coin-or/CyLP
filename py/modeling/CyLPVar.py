import inspect
import numpy as np
from scipy import sparse
from CyLP.py.utils.sparseUtil import csr_matrixPlus, csc_matrixPlus, sparseConcat


class CyLPConstraint:
    def __init__(self):
        self.upper = None
        self.lower = None
        self.vars = []
        self.nRows = -1
        #isRange is true unless we see a sign +, -,*
        self.isRange = True
        #isObjective gets False whenever we see a comparison sign
        self.startRowIndex = -1
        self.isObjective = False 
    def __repr__(self):
        s = ''
        s += '\n_______________________________________\n'
        s += '%d rows:\n'% self.nRows
        s += 'starting at row: %d\n' % self.startRowIndex
        s += str(self.vars)
        s += '\n'
        if self.lower != None:
            s += '\nlowerbound = \n ' + str(self.lower) 
        if self.upper != None:
            s += '\nupperbound = \n ' + str(self.upper)
            
        s += '\n_______________________________________\n'
        return s


def getCoinInfinity():
    return 1.79769313486e+308

class CyLPVar:
        
    currentLineNumber = -1
    currentConstraintIndex = -1
    currentAutoVarIndex = 1
    totalNumberOfConstraints = 0
    constraintList = {}
    
    #this is only made on when we call a function of min, max, primal, ... type
    objectiveMode = False

    def __init__(self, dim, varName='', model = None, isInt = False):
        if model != None:
            self.model = model
            
        # If no name is provided make one
        if varName == '':
            #first try to do something clever
            try:
                callerLine = inspect.getframeinfo(inspect.currentframe().f_back)[3][0]
                self.varName = callerLine[:(callerLine.index('='))].strip()
            except :
                self.varName = 'x_' + str(CyLPVar.currentAutoVarIndex)
                CyLPVar.currentAutoVarIndex += 1
        else:
            self.varName = varName
        
        self.dim = dim
        self.isInt = isInt

        self.rightCoef = {}
        self.leftCoef = {}
        
        self.upper = getCoinInfinity() * CyLPArray(np.ones(self.dim))
        self.lower = -getCoinInfinity() * CyLPArray(np.ones(self.dim))
        
        #set this if we have a slice 
        self.parentVar = None
        self.partialRanges = []
        
        #If not set means all indices
        self.indices = np.arange(0, self.dim)
        
        self.objectiveCoef = np.zeros(self.dim)

    def sum(self):
        self.objectiveCoef = np.ones(self.dim)
        return self

    
    @property
    def shape(self):
        return (self.dim, )
        
    @property
    def constraints(self):
        if self.model != None:
            return CyLPVar.constraintList[self.model]
        else:
            raise Exception('No model set for variable %s' % self.varName)
    
    def addToConstraint(self, currentConstraintInd):
        for var in self.constraints[currentConstraintInd].vars:
            if self.varName == var.varName:
                #print 'Warning: Variable ' + self.varName + ' already in the constraint'
                return
        self.constraints[currentConstraintInd].vars.append(self)
        #CyLPVar.vars[currentConstraintInd].append(self)
        
    
    def __getitem__(self, key):
        if type(key) == slice:
            dim = key.stop - key.start
            newObj = CyLPVar(dim, self.varName + '[' + str(key.start) + ':' + str(key.stop) + ']')
            newObj.indices = np.arange(key.start, key.stop)
            newObj.parentVar = self
            newObj.model = self.model
            return newObj
        elif type(key) == int:
            dim = 1
            newObj = CyLPVar(dim, self.varName + '[' + str(key)  + ']')
            newObj.indices = np.array([key])
            newObj.parentVar = self
            newObj.model = self.model
            const = self.getConstraintIndex(getCallerLineNo())
            newObj.leftCoef[const] = np.array([1])  
            newObj.addToConstraint(const) 
            return newObj
        else:
            raise IndexError(key)
        
    
    def getConstraintIndex(self, callerLineNumber):
        '''
        The current index depends one the line number of the caller. If the line has changed
        it means we have a different constraint. We do this to be able to add constraints
        without a method call. like:
        > u <= Ax <= b
        > x >= 0
        '''
        if callerLineNumber == CyLPVar.currentLineNumber:
                return CyLPVar.currentConstraintIndex
        
        #see if the just finished constraint was a range constraint and remove it
        if len(self.constraints) > 0 and self.getCurrentConstraint().isRange:
            self.constraints.pop()
        elif (len(self.constraints) > 0 and
                                self.getCurrentConstraint().isObjective):
            self.constraints.pop()
        else:
            if len(self.constraints) > 0:
                CyLPVar.totalNumberOfConstraints += self.getCurrentConstraint().nRows
            CyLPVar.currentConstraintIndex += 1
        
        #are we defining an objective?
        frame = inspect.currentframe().f_back.f_back.f_back
        constraintToAdd = CyLPConstraint()
        if  frame != None:
            callerLine = inspect.getframeinfo(frame)[3][0]
            if '==' in callerLine or '>=' in callerLine or '<=' in callerLine:
                CyLPVar.objectiveMode = False
            else:
                CyLPVar.objectiveMode = True
                constraintToAdd.isObjective = True
        else:
            CyLPVar.objectiveMode = False
        self.constraints.append(constraintToAdd)
        CyLPVar.currentLineNumber = callerLineNumber
        self.getCurrentConstraint().startRowIndex = CyLPVar.totalNumberOfConstraints

        #CyLPVar.vars[CyLPVar.currentConstraintIndex] = []
        return CyLPVar.currentConstraintIndex

    def getCurrentConstraint(self):
        return self.constraints[CyLPVar.currentConstraintIndex]
        
    def checkCoefficientDim(self, callerLineNumber, coef):
        key = self.getConstraintIndex(callerLineNumber)
        
        if isnumber(coef):
            nrows = nvars = 1
        else:
            if len(coef.shape) == 1:
                nrows, nvars = 1, coef.shape[0]
            else:
                nrows, nvars = coef.shape
            
        cons = self.getCurrentConstraint()
        #Is it the first time we learn the number rows of the constraint
        if cons.nRows == -1:
            cons.nRows = nrows
            cons.upper = getCoinInfinity() * CyLPArray(np.ones(nrows))
            cons.lower = -getCoinInfinity() * CyLPArray(np.ones(nrows))
        elif cons.nRows != nrows and not isnumber(coef):
            raise ValueError(str(coef) + "\n not the same number of rows")
        
        if nvars != self.dim and not isnumber(coef):
            raise ValueError(str(coef) + "\n not the same number of columns")
        
    def __repr__(self):
        #s = str(self.tokens)
        s = '\n'
        s += self.varName
        if self.upper != None:
            s += '\nupperbound : \n' + str(self.upper)
        if self.lower != None:
            s += '\nlowerbound : \n' + str(self.lower)
        
        if len(self.partialRanges) > 0:
            s += '\npartial ranges: '
            s += str(self.partialRanges)
        
        s += '\n'
            
        if self.rightCoef != None:
                s += 'right coefficient : ' + str(self.rightCoef) + '\n'
        if self.rightCoef != None:
                s += 'left coefficient : ' + str(self.leftCoef) + '\n'
                
        s += '\n'
        return s
    
    def __rmul__(self, other):
        self.checkCoefficientDim(getCallerLineNo(), other)
        key = self.getConstraintIndex(getCallerLineNo())
        self.getCurrentConstraint().isRange = False
        if isnumber(other):
            other = other * np.ones(self.dim)
        if CyLPVar.objectiveMode:
            self.objectiveCoef = other
        else:
            self.leftCoef[key] = other
            self.addToConstraint(key)
        return self
        #return CyLPVar('*', other, self)
    
    def __mul__(self, other):
        raise Exception('right coefficient not supported')
        self.checkCoefficientDim(getCallerLineNo(), other)
        key = self.getConstraintIndex(getCallerLineNo())
        self.rightCoef[key] = other
        self.addToConstraint(key)
        return self
        #return CyLPVar('*', self, other)
    
    def __radd__(self, other):
        key = self.getConstraintIndex(getCallerLineNo())
        cons = self.getCurrentConstraint()
        cons.isRange = False
        
        if CyLPVar.objectiveMode:
            thingToChange = self.objectiveCoef
        else:
            if key in self.leftCoef.keys():
                thingToChange = self.leftCoef[key]
            else:
                if self.dim == other.dim:
                    self.leftCoef[key] = sparse.identity(cons.nRows).todense()
                    thingToChange = self.leftCoef[key]
                else:
                    raise ValueError("dimensions don't match")      
            
        if key not in self.leftCoef.keys():
            if cons.nRows == 1:
                thingToChange = 1
            else:
                thingToChange = sparse.identity(cons.nRows).todense()
        
        if not CyLPVar.objectiveMode:
            self.addToConstraint(key)
        return self
    
    #def __add__(self, other):
    #   return CyLPVar('+', self, other)
    
    def __neg__(self):
        key = self.getConstraintIndex(getCallerLineNo())
        cons = self.getCurrentConstraint()
        cons.isRange = False
        
        if CyLPVar.objectiveMode:
            thingToChange = self.objectiveCoef
        else:
            if key in self.leftCoef.keys():
                thingToChange = self.leftCoef[key]
            else:
                if self.dim == other.dim:
                    self.leftCoef[key] = sparse.identity(cons.nRows).todense()
                    thingToChange = self.leftCoef[key]
                else:
                    raise ValueError("dimensions don't match")
        
        if key not in self.leftCoef.keys():
            if cons.nRows == 1:
                thingToChange = -1
            else:
                thingToChange = -sparse.identity(cons.nRows).todense()
        thingToChange *= -1
        if not CyLPVar.objectiveMode:
            self.addToConstraint(key)
        return self
    
    def __rsub__(self, other):
        key = self.getConstraintIndex(getCallerLineNo())
        cons = self.getCurrentConstraint()
       
        cons.isRange = False

        if CyLPVar.objectiveMode:
            thingToChange = self.objectiveCoef
        else:
            if key in self.leftCoef.keys():
                thingToChange = self.leftCoef[key]
            else:
                if self.dim == other.dim:
                    self.leftCoef[key] = sparse.identity(cons.nRows).todense()
                    thingToChange = self.leftCoef[key]
                else:
                    raise ValueError("dimensions don't match")
            
    
        if key not in self.leftCoef.keys() and not self.objectiveMode:
            if cons.nRows == 1:
                thingToChange = 1
            else:
                thingToChange = sparse.identity(cons.nRows).todense()
        thingToChange *= -1
        
        if not CyLPVar.objectiveMode:
            self.addToConstraint(key)
        return self
        #return CyLPVar('-', other, self)
    
    #def __sub__(self, other):
    #   return CyLPVar('-', self, other)
    
    def __le__(self, other):
        key = self.getConstraintIndex(getCallerLineNo())
        
        # see if there has been no coefficient, means variable range, bound,....
        cons = self.getCurrentConstraint()
        if cons.isRange:
            if self.parentVar != None:
                self.parentVar.upper[self.indices] = other
                found = False
                for p in self.parentVar.partialRanges:
                    if p.varName == self.varName:
                        found = True    
                if not found :
                    self.parentVar.partialRanges.append(self)
            else:
                if isnumber(other):
                    self.upper = other * np.ones((self.dim))
                    #if self.lower == None:
                    #    self.lower = -getInfinity() * np.ones((self.dim))
                else:
                    self.upper = other
                    #if self.lower == None:
                    #    self.lower = -getInfinity() * other.shape[1]
                    
            return self
            #cons.nRows = self.dim
            #self.constraintDim.append(self.dim)
            #self.addToConstraint(key)
        
        if cons.nRows == -1 and not cons.isRange:
            cons.nRows = 1
        if isnumber(other):
            other = other * CyLPArray(np.ones(cons.nRows))
        elif len(other) !=  cons.nRows:
            raise ValueError("Upperbound dimension doesn't match")
        
        cons.upper = other
        if cons.lower == None:
           # if isnumber(other):
            cons.lower = -getCoinInfinity() * CyLPArray(np.ones(cons.nRows))
           # else:
           #     cons.lower = -getInfinity() * CyLPArray(np.ones(cons.nRows))
        #CyLPVar.upper[key] = other
        
        
        return self #CyLPVar('<=', self, other) 
    
    def __ge__(self, other):
        key = self.getConstraintIndex(getCallerLineNo())

        # see if there has been no coefficient, means variable range, bound,....
        cons = self.getCurrentConstraint()
        if cons.isRange:
            if self.parentVar != None:
                self.parentVar.lower[self.indices] = other
                found = False
                for p in self.parentVar.partialRanges:
                    if p.varName == self.varName:
                        found = True    
                if not found :
                    self.parentVar.partialRanges.append(self)
            
            else:
                if isnumber(other):
                    self.lower = other * np.ones((self.dim))
                    #if self.upper == None:
                    #    self.upper = getInfinity() * np.ones((self.dim))
                else:
                    self.lower = other
                    #if self.upper == None:
                    #    print '====================>', other.shape[1]
                    #    self.upper = getInfinity() * other.shape[1]
            return self
        if cons.nRows == -1 and not cons.isRange:
            cons.nRows = 1
        if isnumber(other):
            other = other * CyLPArray(np.ones(cons.nRows))
        elif len(other) != cons.nRows:
            raise ValueError("Lowerbound dimension doesn't match")
                
        cons.lower = other
        if cons.upper == None:
            #if isnumber(other):
            cons.upper = getCoinInfinity() * CyLPArray(np.ones(cons.nRows))
        #CyLPVar.lower[key] = other
        return self #CyLPVar('>=', self, other) 
    
    def __eq__(self, other):
        self.eq = other
        key = self.getConstraintIndex(getCallerLineNo())
        
        # see if there has been no coefficient, means variable range, bound,....
        cons = self.getCurrentConstraint()
        if cons.isRange:
            self.upper = self.lower = other
            if self.parentVar != None:
                self.parentVar.lower[self.indices] = other
                self.parentVar.upper[self.indices] = other
                #found = False
                #for p in self.parentVar.partialRanges:
                #   if p.varName == self.varName:
                #       found = True    
                #if not found :
                #   self.parentVar.partialRanges.append(self)
            else:
                if isnumber(other):
                    self.lower = other * np.ones((self.dim))
                    self.upper = other * np.ones((self.dim))
                else:
                    self.lower = other  
                    self.upper = other
            return self
            
        if isnumber(other):
            other = other * CyLPArray(np.ones(cons.nRows))
        elif len(other) != cons.nRows:
            raise ValueError("Equality: dimension doesn't match")
            
        cons.lower = cons.upper = other
        #CyLPVar.lower[key] = other
        #CyLPVar.upper[key] = other
        return self #CyLPVar('==', self, other)
        
    
    #def __lt__(self, other):
    #   pass #raise NotImplementedError('Modeling does not support < or >')
    
    #def __ne__(self, other):
        pass #raise NotImplementedError('Modeling does not support !=')
    
    #def __gt__(self, other):
    #   pass #raise NotImplementedError('Modeling does not support < or >')

    def makeTheMatrix(self):
        currentVarIndex = 0
        #mat = sparse.coo_matrix((1, 1))
        mat = None
        
        #check if the last constriant was a range, remove it
        if len(self.constraints) > 0 and self.constraints[-1].isRange:
            self.constraints.pop()
        #if len(self.constraints) > 0 and self.constraints[-1].isObjective:
        #    self.constraints.pop()

        
        for i in xrange(len(self.constraints)):
            cons = self.constraints[i]
            consHasVar = False
            for var in cons.vars:
                if self.varName == var.varName :
                    mat = sparseConcat(mat, var.leftCoef[i], 'v')
                    consHasVar = True
                    break
                if var.parentVar != None and self.varName == var.parentVar.varName:
                    #print 'dim = ', var.parentVar.dim
                    zz = csc_matrixPlus((1,var.parentVar.dim))
                    for j in var.indices:
                        zz[0,j] = var.leftCoef[i][0]
                    #print mat.todense()
                    #print var.leftCoef
                    #print 'zz= ', zz
                    #print zz.todense()
                    mat = sparseConcat(mat, zz, 'v')
                    consHasVar = True
                    break
                    
            if not consHasVar:
                #print 'here nnz = ', mat.nnz
                #if mat.nnz == 0:
                #   zz = np.zeros((cons.nRows, self.dim))
                #   mat = sparseConcat(mat, zz, 'v')
                    #mat = zz
                    #mat.addRows(cons.nRows)
                #else:
                    zz = sparse.coo_matrix((cons.nRows, self.dim))
                    #print 'mat'
                    #print mat.todense()
                    #print 'zz'
                    #print zz.todense()
                    #print cons.nRows, self.dim
                    mat = sparseConcat(mat, zz, 'v')
                
        return mat  

    def __lshift__(self, other):
        print 'lshift'
    def __rshift__(self, other):
        print 'rshift'
    
    #@staticmethod
    #def makeTheMatrix():
    #   currentVarIndex = 0
    #   for i in xrange(len(CyLPVar.constraintList)):
    #       print 'constraint %d:' % i
    #       cons = CyLPVar.constraintList[i]
    #       for var in cons.vars:
    #           print var.leftCoef[i]
        
        
#def makeTheMatrix(*vars):
    
#   for var in vars:
        

class CyLPArray(np.ndarray):
    
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.__le__ = CyLPArray.__le__
        obj.__ge__ = CyLPArray.__ge__
        return obj
    
    def __le__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__le__(self, other)
    
    def __ge__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__ge__(self, other)
        
    def __mul__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__mul__(self, other)
    
    def __rmul__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__rmul__(self, other)
    
    def __add__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__add__(self, other)
        
    def __radd__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__radd__(self, other)
        
    def __rsub__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__rsub__(self, other)

    def __sub__(self, other):
        if (issubclass(other.__class__, CyLPVar)):
            return NotImplemented
        return np.ndarray.__sub__(self, other)
    
def getCallerLineNo():
        '''
        If you call this from a function f(), it returns the line number 
        of the python source code that called f
        '''
        return inspect.currentframe().f_back.f_back.f_lineno

def isnumber(x):
    return isinstance(x, (int, long, float))




