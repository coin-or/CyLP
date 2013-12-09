'''
As a part of ``cylp.python.pivots`` it implements Dantzig's
 Simplex dual pivot rule. Although it already exists in CLP,
for testing purposes we implement one in Python.
'''

import sys
import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from DualPivotPythonBase import DualPivotPythonBase
#from cylp.py.pivots import DantzigPivot


class DualDantzigPivot(DualPivotPythonBase):
    '''
    Dantzig's dual pivot rule implementation.

    .. _custom-dual-pivot-usage:

    **Usage**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.pivots.DualDantzigPivot import DualDantzigPivot
    >>> from cylp.py.pivots.DualDantzigPivot import getMpsExample
    >>> # Get the path to a sample mps file
    >>> f = getMpsExample()
    >>> s = CyClpSimplex()
    >>> s.readMps(f)  # Returns 0 if OK
    0
    >>> pivot = DualDantzigPivot(s)
    >>> s.setDualPivotMethod(pivot)
    >>> s.dual()
    'optimal'
    >>> round(s.objectiveValue, 5)
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
            return np.argmax(infeasibilities)
        return -1

    def updateWeights(self, inp, spare, spare2, updatedColumn):
        model = self.clpModel
        pr = model.pivotRow()
        model.updateColumnFT(spare, updatedColumn)
        indices = updatedColumn.indices
        elements = updatedColumn.elements
        if updatedColumn.isInPackedMode:
            if pr in indices:
                ind = np.where(indices==pr)[0][0]
                return elements[ind]
        else:
            return elements[pr]
        return 0

    def updatePrimalSolution(self, primalUpdate, primalRatio, objectiveChange):
        model = self.clpModel
        nConstraints = model.nConstraints
        basicVarInds = model.basicVariables
        rowNumbers = primalUpdate.indices
        elements = primalUpdate.elements
        nElements = primalUpdate.nElements
        changeObj = 0

        sol = model.solution[basicVarInds[rowNumbers]]
        cost = model.cost[basicVarInds[rowNumbers]]

        if primalUpdate.isInPackedMode:
            change = primalRatio * elements[:nElements]
            model.solution[basicVarInds[rowNumbers]] -= change
        else:
            change = primalRatio * elements[rowNumbers]
            model.solution[basicVarInds[rowNumbers]] -= change

        changeObj = -np.dot(change, cost)
        primalUpdate.clear()
        objectiveChange[0] += changeObj

        return changeObj


def getMpsExample():
    import os
    import inspect
    import sys
    cylpDir = os.environ['CYLP_SOURCE_DIR']
    return os.path.join(cylpDir, 'cylp', 'input', 'p0033.mps')


if __name__ == "__main__":
    print sys.argv
    if len(sys.argv) == 1:
        import doctest
        doctest.testmod()
    else:
        from cylp.cy import CyClpSimplex
        from cylp.py.pivots import DualDantzigPivot
        s = CyClpSimplex()
        s.readMps(sys.argv[1])  # Returns 0 if OK
        pivot = DualDantzigPivot(s)
        s.setDualPivotMethod(pivot)
        s.dual()
