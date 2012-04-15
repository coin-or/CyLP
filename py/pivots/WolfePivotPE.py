from itertools import izip
import random
import numpy as np
from CyLP.cy import CyCoinIndexedVector
from CyLP.cy.CyClpSimplex import cydot
from PivotPythonBase import PivotPythonBase


class WolfePivotPE(PivotPythonBase):

    def __init__(self, clpModel, bucketSize=1):
        self.dim = clpModel.nConstraints + clpModel.nVariables
        self.clpModel = clpModel
        # Require extra check after leaving variable is chosen
        clpModel.useCustomPrimal(True)  
        #self.banList = np.array([])
        self.orgBan = np.array(self.dim * [True], np.bool)
        self.notBanned = self.orgBan.copy()
        self.complementarityList = np.arange(self.dim)


        #Positive-edge-related attributes
        self.isDegenerate = False

        # Create some numpy arrays here ONCE to prevent memory
        # allocation at each iteration
        self.aColumn = CyCoinIndexedVector()
        self.aColumn.reserve(self.dim)
        self.w = CyCoinIndexedVector()
        self.w.reserve(self.clpModel.nRows)

        self.rhs = np.empty(self.clpModel.nRows, dtype=np.double)
        self.EPSILON = 10**-7
        self.lastUpdateIteration = 0
        
        self.compCount = 0
        self.nonCompCount = 0
        self.compRej = 0

    # Begining of Positive-Edge-related attributes
    
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

        #print 'degeneracy level : ', (len(self.z)) / float(nRows)
        #self.isDegenerate = (len(self.p) != nRows)
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

    # End of Positive-Edge-related attributes

    def pivotColumn(self):
        s = self.clpModel
        
        self.CompIter = True

        rc = s.reducedCosts
        #tol = s.dualTolerance()
        tol = 0
        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree) & 
                                     self.notBanned)[0]
        
        rc2 = abs(rc[indicesToConsider])
        
        checkFree = False
        
        maxRc = maxCompRc = maxInd = maxCompInd = -1
        

        if self.isDegenerate:
            w = self.w.elements
            compatibility = np.zeros(s.nCols + s.nRows, dtype=np.double)
            if len(indicesToConsider) > 0:
                s.transposeTimesSubsetAll(indicesToConsider,
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
            self.compCount += 1
            return maxCompInd
        self.nonCompCount += 1
        self.CompIter = False
        self.updateW()
        return maxInd
        
#        
#        
#        if rc2.shape[0] > 0:
#            if checkFree:
#                w = np.where(s.varIsFree)[0]
#                if w.shape[0] > 0:
#                    ind = s.argWeightedMax(rc2, indicesToConsider, 1, w)
#                else:
#                    ind = np.argmax(rc2)
#            else:
#                    ind = np.argmax(rc2)
#            #print 'incomming var: %d' % indicesToConsider[ind]
#            ret = indicesToConsider[ind] 
#            del indicesToConsider  # not sure if this is necessary
#            del rc2  # HUGE memory leak otherwise
#            return ret 
#        return -1
#        return self.pivotColumnFirst()


    def isPivotAcceptable(self):
        #import pdb; pdb.set_trace() 
#        return True
#        #TODO ComplementarityList can be defined in the current class
        s = self.clpModel
        #cl = s.getComplementarityList()
        cl = self.complementarityList
        pivotRow = s.pivotRow()
        if pivotRow < 0:
            colInd = s.sequenceIn()
            #print 'entering: ', colInd, ' comp: ', cl[colInd]
            return 1

        pivotVariable = s.getPivotVariable()
        leavingVarIndex = pivotVariable[pivotRow]
        colInd = s.sequenceIn()

#        print 'Basis:'
#        print s.getPivotVariable()
#        print 'leave: ', leavingVarIndex
#        print 'entering: ', colInd, ' comp: ', cl[colInd]
         
        if s.getVarStatus(cl[colInd]) == 1 and \
            cl[colInd] != leavingVarIndex:
            #self.banList = np.concatenate((self.banList, [colInd]))
            self.notBanned[colInd] = False
            #s.setFlagged(colInd)
            if self.CompIter:
                self.compRej += 1
            return 0

        
        #self.banList = np.zeros(self.dim, np.int)
        del self.notBanned
        self.notBanned = self.orgBan.copy()
        #self.notBanned = np.array(self.dim * [True])

        return 1
    
    def setComplement(self, model, v1, v2):
        v1n = v1.name
        v2n = v2.name
        listv1 = model.inds.varIndex[v1n]
        listv2 = model.inds.varIndex[v2n]
        for i, j in izip(listv1, listv2):
            (self.complementarityList[i], self.complementarityList[j]) = \
             (self.complementarityList[j], self.complementarityList[i])

