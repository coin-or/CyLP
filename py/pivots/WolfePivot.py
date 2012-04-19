from itertools import izip
import numpy as np
from PivotPythonBase import PivotPythonBase


class WolfePivot(PivotPythonBase):

    def __init__(self, clpModel, bucketSize=1):
        self.dim = clpModel.nConstraints + clpModel.nVariables
        self.clpModel = clpModel
        # Tell IClpSimplex that this pivot rules needs 
        #an extra check after the leaving varible is chosen.
        clpModel.useCustomPrimal(True)
        #self.banList = np.array([])
        self.orgBan = np.array(self.dim * [True], np.bool)
        self.notBanned = self.orgBan.copy()
        self.complementarityList = np.arange(self.dim)
    
    
    def pivotColumn(self):
        s = self.clpModel
        
        # If objective function linear proceed as normal 
#        if s.Hessian == None:
        rc = s.reducedCosts
#        print 'u:', s.varIsAtUpperBound
#        print 'l:', s.varIsAtLowerBound
#        print 'f:', s.varIsFlagged
#        print 'free:', s.varIsFree
#        print 'ban:', self.notBanned
#        print 'bas', s.varIsBasic
        #tol = s.dualTolerance()
        tol = 0
        #print 'Basis:'
        #print s.getPivotVariable()
#        iii = np.where(s.varNotFlagged & s.varNotFixed &
#                                     s.varNotBasic &
#                                     (((rc > tol) & s.varIsAtUpperBound) |
#                                     ((rc < -tol) & s.varIsAtLowerBound) |
#                                     s.varIsFree))[0] 
        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree) & 
                                     self.notBanned)[0]

        #print iii
        #print 'rc = ', rc[iii]
        #for ii in range(s.nVariables, s.nVariables + s.nConstraints):
        #    if rc[ii] < -tol:
        #        indicesToConsider = np.concatenate((indicesToConsider, [ii]))
        
        #freeVarInds = np.where(s.varIsFree)
        #rc[freeVarInds] *= 10
        
        #print 'before ', indicesToConsider
        #print '~~~~~~~~~~~~~self.banList:', self.banList 
        #np.delete(indicesToConsider, self.banList)
        #print 'after  ', indicesToConsider
        #import pdb; pdb.set_trace() 
        
        rc2 = abs(rc[indicesToConsider])
        
        checkFree = False
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
            #print 'incomming var: %d' % indicesToConsider[ind]
            ret = indicesToConsider[ind] 
            del indicesToConsider  # not sure if this is necessary
            del rc2  # HUGE memory leak otherwise
            return ret 
        return -1
        return self.pivotColumnFirst()


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
            #print 'pivotRow < 0'
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
            #print colInd , ' flagged'
            #self.banList[colInd] = 1
            #print 'banning %d' % colInd
            #print self.notBanned
            #self.banList = np.concatenate((self.banList, [colInd]))
            self.notBanned[colInd] = False
            #print self.notBanned
            #s.setFlagged(colInd)
            return 0

        
        #self.banList = np.zeros(self.dim, np.int)
        #print "reseting>>>>>>>>>>>>>>>>>>>>>>>"
        del self.notBanned
        self.notBanned = self.orgBan.copy()
        #self.notBanned = np.array(self.dim * [True])

        return 1
    
#    def setComplement(self, list1, list2):
#        for i, j in izip(list1, list2):
#            (self.complementarityList[i], self.complementarityList[j]) = \
#             (self.complementarityList[j], self.complementarityList[i])
    
    def setComplement(self, model, v1, v2):
        v1n = v1.name
        v2n = v2.name
        listv1 = np.array(model.inds.varIndex[v1n])[v1.indices]
        listv2 = np.array(model.inds.varIndex[v2n])[v2.indices]
        for i, j in izip(listv1, listv2):
            self.complementarityList[i], self.complementarityList[j] = j, i

