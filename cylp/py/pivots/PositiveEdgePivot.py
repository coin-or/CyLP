'''
As a part of ``cylp.python.pivots`` it implements the positive edge
pivot selection rule.
'''

from __future__ import print_function
import random
import numpy as np
from cylp.cy import CyCoinIndexedVector
from cylp.cy.CyClpSimplex import cydot
from .PivotPythonBase import PivotPythonBase


class PositiveEdgePivot(PivotPythonBase):
    '''
    Positive Edge pivot rule implementation.

    .. _custom-pivot-usage:

    **Usage**

    >>> from cylp.cy import CyClpSimplex
    >>> from cylp.py.pivots import PositiveEdgePivot
    >>> from cylp.py.pivots.PositiveEdgePivot import getMpsExample
    >>> # Get the path to a sample mps file
    >>> f = getMpsExample()
    >>> s = CyClpSimplex()
    >>> s.readMps(f)  # Returns 0 if OK
    0
    >>> pivot = PositiveEdgePivot(s)
    >>> s.setPivotMethod(pivot)
    >>> s.primal()
    'optimal'
    >>> round(s.objectiveValue, 5)
    2520.57174

    '''

    def __init__(self, clpModel, EPSILON=10 ** (-7)):
        self.clpModel = clpModel
        self.dim = self.clpModel.nRows + self.clpModel.nCols

        self.isDegenerate = False

        # Create some numpy arrays here ONCE to prevent memory
        # allocation at each iteration
        self.aColumn = CyCoinIndexedVector()
        self.aColumn.reserve(self.dim)
        self.w = CyCoinIndexedVector()
        self.w.reserve(self.clpModel.nRows)

        self.rhs = np.empty(self.clpModel.nRows, dtype=np.double)
        self.EPSILON = EPSILON
        self.lastUpdateIteration = 0

    def updateP(self):
        '''Finds constraints with abs(rhs) <=  epsilon and put
        their indices in "z"
        '''
        s = self.clpModel
        nRows = s.nRows

        rhs = self.rhs
        s.getRightHandSide(rhs)

        #self.p = np.where(np.abs(rhs) > self.EPSILON)[0]
        self.z = np.where(np.abs(rhs) <= self.EPSILON)[0]

        #print('degeneracy level : ', (len(self.z)) / float(nRows)))
        self.isDegenerate = (len(self.z) > 0)

    def updateW(self):
        '''Sets "w" to be a vector of random vars with "0"
        at indices defined in "p"
        Note that vectorTimesB_1 changes "w"
        '''
        self.updateP()
        self.w.clear()
        self.w[self.z] = np.random.random(len(self.z))
        s = self.clpModel
        s.vectorTimesB_1(self.w)

        self.lastUpdateIteration = s.iteration

    def random(self):
        'Defines how random vector "w" components are generated'
        return random.random()

    def isCompatible(self, varInd):
        if not self.isDegenerate:
            return False
        s = self.clpModel
        s.getACol(varInd, self.aColumn)

        return abs(cydot(self.aColumn, self.w)) < self.EPSILON

    def checkVar(self, i):
        return self.isCompatible(i)

    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.reducedCosts

        tol = s.dualTolerance
        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        rc2 = abs(rc[indicesToConsider])

        maxRc = maxCompRc = maxInd = maxCompInd = -1

        if self.isDegenerate:
            w = self.w.elements
            compatibility = np.zeros(s.nCols + s.nRows, dtype=np.double)
            if len(indicesToConsider) > 0:
                s.transposeTimesSubsetAll(indicesToConsider.astype(np.int64),
                                          w, compatibility)
            comp_varInds = indicesToConsider[np.where(abs(
                                    compatibility[indicesToConsider]) <
                                    self.EPSILON)[0]]

            comp_rc = abs(rc[comp_varInds])
            if len(comp_rc) > 0:
                maxCompInd = comp_varInds[np.argmax(comp_rc)]
                maxCompRc = rc[maxCompInd]

        if len(rc2) > 0:
            maxInd = indicesToConsider[np.argmax(rc2)]
            maxRc = rc[maxInd]

        del rc2
        if maxCompInd != -1 and abs(maxCompRc) > 0.1 * abs(maxRc):
            return maxCompInd
        self.updateW()
        return maxInd

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        return True


def getMpsExample():
    import os
    import inspect
    cylpDir = os.environ['CYLP_SOURCE_DIR']
    return os.path.join(cylpDir, 'cylp', 'input', 'p0033.mps')

