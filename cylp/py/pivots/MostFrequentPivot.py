import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from .PivotPythonBase import PivotPythonBase


class MostFrequentPivot(PivotPythonBase):
    '''
    Last-In-First-Out pivot rule implementation.

    **Usage**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.pivots import MostFrequentPivot
    >>> from cylp.py.pivots.MostFrequentPivot import getMpsExample
    >>> # Get the path to a sample mps file
    >>> f = getMpsExample()
    >>> s = CyClpSimplex()
    >>> s.readMps(f)  # Returns 0 if OK
    0
    >>> pivot = MostFrequentPivot(s)
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
        self.priorityList = list(range(self.dim))
        self.frequencies = np.zeros(self.dim)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.nRows + s.nCols

        tol = s.dualTolerance

        for i in self.priorityList:
            if s.flagged(i) or s.CLP_getVarStatus(i) == 5:  # flagged or fixed
                continue

            #TODO: can we just say dualInfeasibility = rc[i] ** 2
            if s.CLP_getVarStatus(i) == 2:  # upperbound
                dualInfeasibility = rc[i]
            # free or superbasic
            elif s.CLP_getVarStatus(i) == 4 or s.CLP_getVarStatus(i) == 0:
                dualInfeasibility = abs(rc[i])
            else:  # lowerbound
                dualInfeasibility = -rc[i]

            if dualInfeasibility > tol:
                self.addFrequency(i)
                return i

        return -1

    def addFrequency(self, i):
        '''
        Add one to frequency of variable i,
        resorts the priorityList (always sorted)
        '''
        self.frequencies[i] += 1
        self.priorityList.remove(i)
        for j in range(self.dim):
            if self.frequencies[i] >= self.frequencies[self.priorityList[j]]:
                self.priorityList.insert(j, i)
                return
        self.priorityList.append(i)

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True


def getMpsExample():
    import os
    import inspect
    cylpDir = os.environ['CYLP_SOURCE_DIR']
    return os.path.join(cylpDir, 'cylp', 'input', 'p0033.mps')
