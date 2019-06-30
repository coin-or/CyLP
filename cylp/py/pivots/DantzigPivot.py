'''
As a part of ``cylp.python.pivots`` it implements Dantzig's
Classical Simplex pivot rule. Although it already exists in CLP,
for testing purposes we implement one in Python.
'''

import sys
import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from .PivotPythonBase import PivotPythonBase


class DantzigPivot(PivotPythonBase):
    '''
    Dantzig's pivot rule implementation.

    **Usage**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.pivots import DantzigPivot
    >>> from cylp.py.pivots.DantzigPivot import getMpsExample
    >>> # Get the path to a sample mps file
    >>> f = getMpsExample()
    >>> s = CyClpSimplex()
    >>> s.readMps(f)  # Returns 0 if OK
    0
    >>> pivot = DantzigPivot(s)
    >>> s.setPivotMethod(pivot)
    >>> s.primal()
    'optimal'
    >>> round(s.objectiveValue, 5)
    2520.57174

    '''

    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel

        # Update the reduced costs, for both the original and the slack variables
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        rc = s.reducedCosts
        tol = s.dualTolerance

        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        #freeVarInds = np.where(s.varIsFree)
        #rc[freeVarInds] *= 10

        rc2 = np.abs(rc[indicesToConsider])

        checkFree = True
        #rc2[np.where((status & 7 == 4) | (status & 7 == 0))] *= 10
        if rc2.shape[0] > 0:
            if checkFree:
                w = np.where(s.varIsFree)[0]
                if w.shape[0] > 0:
                    ind = s.argWeightedMax(rc2, indicesToConsider, 1, w)
                else:
                    ind = np.argmax(rc2)
            else:
                    ind = np.argmax(rc2)
            #del rc2
            return  indicesToConsider[ind]
        return -1

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True


def getMpsExample():
    import os
    import inspect
    cylpDir = os.environ['CYLP_SOURCE_DIR']
    return os.path.join(cylpDir, 'cylp', 'input', 'p0033.mps')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        import doctest
        doctest.testmod()
    else:
        from cylp.cy import CyClpSimplex
        from cylp.py.pivots import DantzigPivot
        s = CyClpSimplex()
        s.readMps(sys.argv[1])
        pivot = DantzigPivot(s)
        s.setPivotMethod(pivot)
        s.primal()
