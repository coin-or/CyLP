import numpy as np
from operator import itemgetter
from random import shuffle
from cylp.cy import CyCoinIndexedVector
import random
from math import floor
from cylp.cy.CyClpSimplex import cydot
from cylp.py.pivots import PositiveEdgePivot


class PositiveEdgeWolfePivot(PositiveEdgePivot):

    def __init__(self, clpModel, bucketSize=1, EPSILON=10 ** (-7)):
        PositiveEdgePivot.__init__(self, clpModel, bucketSize, EPSILON)
        self.banList = []

    def pivotColumnFirst(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        '''Finds the first variable with an acceptable
        reduced costs searching in random buckets
        '''
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.getNumRows() + s.getNumCols()
        nRows = s.getNumRows()

        tol = s.dualTolerance()

        if not self.initialized:
            self.initialize()

        xranges = self.xranges
        numberOfSegments = self.numberOfSegments
        segmentSize = self.segmentSize

        shuffle(xranges)

        bestFreeVarInd = -1
        bestVarInd = -1
        bestCompVarInd = -1

        bestFreeVarRC = 0
        bestVarRC = 0
        bestCompVarRC = 0

        goodVariableFound = 0

        for iSegment in xrange(numberOfSegments):
#           comp_ind = [i for i in xrange(dim) if not s.flagged(i) and \
#                     (rc[i] < -tol and \
#                     s.getStatus(i) == 3 and \
#                     self.isCompatible(i)) \
#                     or \
#                     (rc[i] > tol and \
#                     s.getStatus(i) == 2 and \
#                     self.isCompatible(i)) \
#                     or \
#                     (abs(rc[i]) > tol and \
#                     (s.getStatus(i) == 4 or s.getStatus(i) == 0) and \
#                     self.isCompatible(i))]

            for i in xranges[iSegment]:
                if s.flagged(i) or \
                            abs(rc[i]) < self.EPSILON or \
                            i in self.banList:
                    continue
                if s.getStatus(i) == 3:  # at lowerbound
                    if rc[i] < bestCompVarRC and self.isCompatible(i):
                        bestCompVarRC = rc[i]
                        bestCompVarInd = i
                        goodVariableFound += 1
                    if rc[i] < bestVarRC:
                        bestVarRC = rc[i]
                        bestVarInd = i
                        goodVariableFound += 1

                elif s.getStatus(i) == 2:  # at upperbound
                    r = -rc[i]
                    if r < bestCompVarRC and self.isCompatible(i):
                        bestCompVarRC = r
                        bestCompVarInd = i
                        goodVariableFound += 1
                    if r < bestVarRC:
                        bestVarRC = r
                        bestVarInd = i
                        goodVariableFound += 1
                # Free or Superbasic
                elif s.getStatus(i) == 0 or s.getStatus(i) == 4:
                    r = -abs(rc[i])
                    if r < bestCompVarRC and self.isCompatible(i):
                        bestCompVarRC = r
                        bestCompVarInd = i
                        goodVariableFound += 1
                    if r < bestFreeVarRC:
                        bestFreeVarRC = r
                        bestFreeVarInd = i
                        goodVariableFound += 1

            if goodVariableFound > 0:
                break

        l = [bestVarRC, bestFreeVarRC, bestCompVarRC]

        needUpdate = True
        if bestVarInd != -1:
            if bestFreeVarInd != -1:
                if 0.1 * bestVarRC > bestFreeVarRC:
                    bestVarRC = bestFreeVarRC
                    bestVarInd = bestFreeVarInd

            if bestCompVarInd != -1:
                if 0.4 * bestVarRC > bestCompVarRC:
                    bestVarRC = bestCompVarRC
                    bestVarInd = bestCompVarInd
                    needUpdate = False

        elif bestFreeVarInd != -1:
            if bestCompVarInd != -1 and 0.4 * bestFreeVarRC > bestCompVarRC:
                bestVarRC = bestCompVarRC
                bestVarInd = bestCompVarInd
                needUpdate = False
            else:
                bestVarRC = bestFreeVarRC
                bestVarInd = bestFreeVarInd

        if needUpdate:
            self.updateP()
            self.updateW()

        return bestVarInd

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
        #TODO ComplementarityList can be defined in the current class
        s = self.clpModel
        cl = s.getComplementarityList()
        pivotRow = s.pivotRow()
        if pivotRow < 0:
            return 1

        pivotVariable = s.getPivotVariable()
        leavingVarIndex = pivotVariable[pivotRow]
        colInd = s.sequenceIn()

        if s.getStatus(cl[colInd]) == 1 and \
            cl[colInd] != leavingVarIndex:
            self.banList.append(colInd)

            #s.setFlagged(colInd)
            return 0

        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []

        return 1
