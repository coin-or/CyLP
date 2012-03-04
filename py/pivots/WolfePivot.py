import numpy as np
from operator import itemgetter
from random import shuffle
from math import floor
from PivotPythonBase import PivotPythonBase


class WolfePivot(PivotPythonBase):

    def __init__(self, clpModel, bucketSize=1):
        self.dim = clpModel.getNumRows() + clpModel.getNumCols()
        self.clpModel = clpModel
        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []
        self.bucketSize = bucketSize
        self.initialized = False

    def pivotColumn(self):
        'Runs the desired pivotColumn'
        #TODO: this should be controled by an attribute
        return self.pivotColumnFirst()

    def initialize(self):
        'Sets self.xranges containing the buckets to be used in pivotColumn'
        dim = self.dim
        segmentSize = max(1, int(floor(dim * self.bucketSize)))
        print "segmentSize = ", segmentSize
        if dim % segmentSize == 0:
            numberOfSegments = dim / segmentSize
        else:
            numberOfSegments = dim / segmentSize + 1

        self.xranges = [xrange(segment * segmentSize,
                               min((segment + 1) * segmentSize, dim))
                        for segment in xrange(numberOfSegments)]

        self.numberOfSegments = numberOfSegments
        self.segmentSize = segmentSize
        self.initialized = True

    def pivotColumnBest(self):
        'Finds the variable with the best reduced cost and returns its index'
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.getNumRows() + s.getNumCols()

        tol = s.dualTolerance()

        L = [(i, rc[i]) for i in xrange(dim) if rc[i] < -tol and \
                                 not s.flagged(i) and \
                                 #not self.banList[i] and \
                                 i not in self.banList and \
                                 s.getStatus(i) == 3]  # at its lowerbound
        L += [(i, -rc[i]) for i in xrange(dim) if rc[i] > tol and \
                                 not s.flagged(i) and \
                                 #not self.banList[i] and \
                                 i not in self.banList and \
                                 s.getStatus(i) == 2]  # at its upperbound

        free = [(i, -abs(rc[i])) for i in xrange(dim)
                                if abs(rc[i]) > tol and \
                                 not s.flagged(i) and \
                                 #not self.banList[i] and \
                                 i not in self.banList and \
                                 (s.getStatus(i) == 4 or s.getStatus(i) == 0)]

        # No profitable entering variable
        if not L:
            smallest = [-1]
        else:
            # Minimum w.r.t. second component.
            smallest = min(L, key=itemgetter(1))

        if not free:
            smallestFree = [-1]
        else:
            smallestFree = min(free, key=itemgetter(1))

        if 0.1 * rc[smallest[0]] < rc[smallestFree[0]]:
            # Variable with smallest reduced cost.
            return smallest[0]
        return smallestFree[0]

    def pivotColumnFirst(self):
        '''
        Finds the first variable with an acceptable reduced costs
        searching in random buckets
        '''
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.getNumRows() + s.getNumCols()

        tol = s.dualTolerance()

        if not self.initialized:
            self.initialize()

        xranges = self.xranges
        numberOfSegments = self.numberOfSegments
        segmentSize = self.segmentSize

        shuffle(xranges)
        bestFreeVarind = -1

        for segment in xrange(numberOfSegments):
            L = [(i, rc[i]) for i in xranges[segment] if rc[i] < -tol and \
                                     not s.flagged(i) and \
                                     #not self.banList[i] and \
                                     i not in self.banList and \
                                     s.getStatus(i) == 3]  # at its lowerbound
            L += [(i, -rc[i]) for i in xranges[segment] if rc[i] > tol and \
                                     not s.flagged(i) and \
                                     #not self.banList[i] and \
                                     i not in self.banList and \
                                     s.getStatus(i) == 2]  # at its upperbound

            free = [(i, -abs(rc[i])) for i in xranges[segment]
                                if abs(rc[i]) > tol and \
                                 not s.flagged(i) and \
                                 #not self.banList[i] and \
                                 i not in self.banList and \
                                 (s.getStatus(i) == 4 or s.getStatus(i) == 0)]

            # No profitable entering variable
            if not L:
                smallest = -1
            else:
                # Minimum w.r.t. second component.
                smallest = min(L, key=itemgetter(1))[0]

            if not free:
                smallestFree = -1
            else:
                smallestFree = min(free, key=itemgetter(1))[0]

            if  smallest != -1 and smallestFree != -1:
                #bias toward free variables
                if  0.1 * rc[smallest] > rc[smallestFree]:
                    return smallestFree
            if smallestFree != -1:
                return smallestFree
            elif smallest != -1:
                return smallest

        return -1

    def pivotColumnFirst2(self):
        '''
        Finds the first variable with an acceptable reduced costs
        seraching in buckets each time a bucket is chosen
        it is shifted to left
        '''
        s = self.clpModel
        rc = s.getReducedCosts()
        dim = s.getNumRows() + s.getNumCols()

        tol = s.dualTolerance()

        if not self.initialized:
            self.initialize()

        xranges = self.xranges
        numberOfSegments = self.numberOfSegments
        segmentSize = self.segmentSize

        #shuffle(xranges)

        for segment in xrange(numberOfSegments):
            L = [(i, rc[i]) for i in xranges[segment] if rc[i] < -tol and \
                                    not s.flagged(i) and \
                                    #not self.banList[i] and \
                                    i not in self.banList and \
                                    s.getStatus(i) == 3]  # at its lowerbound
            L += [(i, -rc[i]) for i in xranges[segment] if rc[i] > tol and \
                                    not s.flagged(i) and \
                                    #not self.banList[i] and \
                                    i not in self.banList and \
                                    s.getStatus(i) == 2]  # at its upperbound

            free = [(i, -abs(rc[i])) for i in xranges[segment]
                                if abs(rc[i]) > tol and \
                                not s.flagged(i) and \
                                #not self.banList[i] and \
                                i not in self.banList and \
                                (s.getStatus(i) == 4 or s.getStatus(i) == 0)]

            # No profitable entering variable
            if not L:
                smallest = -1
            else:
                # Minimum w.r.t. second component.
                smallest = min(L, key=itemgetter(1))[0]

            if not free:
                smallestFree = -1
            else:
                smallestFree = min(free, key=itemgetter(1))[0]

            if  smallest != -1 and smallestFree != -1:
                #bias toward free variables
                if  0.1 * rc[smallest] > rc[smallestFree]:
                    self.prefer(segment)
                    return smallestFree
            if smallestFree != -1:
                self.prefer(segment)
                return smallestFree
            elif smallest != -1:
                self.prefer(segment)
                return smallest

        return -1

    def prefer(self, segment):
        'Move segment one step to the left'
        if segment != 0:
            self.xranges[segment], self.xranges[segment - 1] = \
                            self.xranges[segment - 1], self.xranges[segment]

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
            #print colInd , ' flagged'
            #self.banList[colInd] = 1
            self.banList.append(colInd)

            #s.setFlagged(colInd)
            return 0

        #self.banList = np.zeros(self.dim, np.int)
        self.banList = []

        return 1
