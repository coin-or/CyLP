import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from .PivotPythonBase import PivotPythonBase
from cylp.cy.CyClpSimplex import VarStatus


class LIFOPivot(PivotPythonBase):
    '''
    Last-In-First-Out pivot rule implementation.

    **Usage**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.pivots import LIFOPivot
    >>> from cylp.py.pivots.LIFOPivot import getMpsExample
    >>> # Get the path to a sample mps file
    >>> f = getMpsExample()
    >>> s = CyClpSimplex()
    >>> s.readMps(f)  # Returns 0 if OK
    0
    >>> pivot = LIFOPivot(s)
    >>> s.setPivotMethod(pivot)
    >>> s.primal()
    'optimal'
    >>> round(s.objectiveValue, 5)
    2520.57174

    '''

    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel
        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []
        self.priorityList = range(self.dim)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.reducedCosts
        dim = s.nCols + s.nRows

        tol = s.dualTolerance

        for i in self.priorityList:
            #flagged or fixed
            if s.flagged(i) or s.CLP_getVarStatus(i) == VarStatus.fixed:
                continue

            #TODO: can we just say dualInfeasibility = rc[i] ** 2
            if s.CLP_getVarStatus(i) == VarStatus.atUpperBound:  # upperbound
                dualInfeasibility = rc[i]
            elif (s.CLP_getVarStatus(i) == VarStatus.superBasic or
                    s.CLP_getVarStatus(i) == VarStatus.free):  # free or superbasic
                dualInfeasibility = abs(rc[i])
            else:  # lowerbound
                dualInfeasibility = -rc[i]

            if dualInfeasibility > tol:
                return i

        return -1

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        '''
        Inserts the leaving variable index as the first element
        in self.priorityList
        '''
        s = self.clpModel

        pivotRow = s.pivotRow()
        if pivotRow < 0:
            return True

        pivotVariable = s.getPivotVariable()
        leavingVarIndex = pivotVariable[pivotRow]

        self.priorityList.remove(leavingVarIndex)
        self.priorityList.insert(0, leavingVarIndex)

        return True


def getMpsExample():
    import os
    import inspect
    cylpDir = os.environ['CYLP_SOURCE_DIR']
    return os.path.join(cylpDir, 'cylp', 'input', 'p0033.mps')
