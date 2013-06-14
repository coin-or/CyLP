'''
As a part of ``CyLP.python.pivots`` it implements Dantzig's
 Simplex dual pivot rule. Although it already exists in CLP,
for testing purposes we implement one in Python.
'''

import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from DualPivotPythonBase import DualPivotPythonBase


class DualDantzigPivot(DualPivotPythonBase):
    '''
    Dantzig's dual pivot rule implementation.

    **Usage**

    from CyLP.cy import CyClpSimplex
    from CyLP.py.pivots.DualDantzigPivot import DualDantzigPivot
    from CyLP.py.pivots.DualDantzigPivot import getMpsExample
    # Get the path to a sample mps file
    f = getMpsExample()
    s = CyClpSimplex()
    s.readMps(f)  # Returns 0 if OK
    0
    pivot = DualDantzigPivot(s)
    s.setDualPivotMethod(pivot)
    s.dual()
    'optimal'
    round(s.objectiveValue, 5)
    2520.57174

    '''

    def __init__(self, clpModel):
        self.dim = clpModel.nRows + clpModel.nCols
        self.clpModel = clpModel

    def pivotRow(self):
        model = self.clpModel
        nConstraints = model.nConstraints
        basicVarInds = model.basicVariables

        u = model.upper[basicVarInds]
        l = model.lower[basicVarInds]
        s = model.solution[basicVarInds]
        infeasibilities = np.maximum(s - u, l - s)

        m = max(infeasibilities)

        if m > model.primalTolerance:
            #print 'returning: ', model.iteration, np.argmax(infeasibilities)
            return np.argmax(infeasibilities)
        #print 'returning -1'
        return -1

    def updateWeights(self, inp, spare, spare2, updatedColumn):
        print '----------------------> '
        model = self.clpModel
        pr = model.pivotRow()
        model.updateColumnFT(spare, updatedColumn)
        updatedColumn.Print()
        indices = updatedColumn.indices
        elements = updatedColumn.elements
        print updatedColumn.nElements
        print updatedColumn.isInPackedMode
        if updatedColumn.isInPackedMode:
            if pr in indices:
                ind = np.where(indices==pr)[0][0]
                print 'pr: ', pr, ', alpha = ', elements[ind]
                return elements[ind]
        else:
            return elements[pr]
        return 0

    def updatePrimalSolution(self, inp, theta):
        print 'updddddddddddddddddddddddatePrimalSolution'
        return

    def pivotColumnStatusWhere(self):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel
        rc = s.reducedCosts

        tol = s.dualTolerance
        #tol = 0
        #incides of vars not fixed and not flagged
        #indicesToConsider = np.where((status & 7 != 1) & (status & 7 != 5) &
        #        (status & 64 == 0) & (((rc > tol) & (status & 7 == 2)) |
        #            ((rc < -tol) & (status & 7 == 3))) )[0]

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


def getMpsExample():
    import os
    import inspect
    curpath = os.path.dirname(inspect.getfile(inspect.currentframe()))
    return os.path.join(curpath, '../../input/p0033.mps')

if __name__ == "__main__":
    from CyLP.cy import CyClpSimplex
    from CyLP.py.pivots.DualDantzigPivot import DualDantzigPivot
    from CyLP.py.pivots.DualDantzigPivot import getMpsExample
    # Get the path to a sample mps file
    f = getMpsExample()
    s = CyClpSimplex()
    s.readMps(f)  # Returns 0 if OK
    pivot = DualDantzigPivot(s)
    s.setDualPivotMethod(pivot)
    s.dual()
    #import doctest
    #doctest.testmod()
