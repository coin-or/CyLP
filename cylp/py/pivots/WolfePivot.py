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
        self.orgBan = np.array(self.dim * [True], np.bool)
        self.notBanned = self.orgBan.copy()
        self.complementarityList = np.arange(self.dim)


    def pivotColumn(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        self.updateReducedCosts(updates, spareRow1, spareRow2, spareCol1, spareCol2)
        s = self.clpModel

        # If objective function linear proceed as normal
#        if s.Hessian == None:
        rc = s.reducedCosts
        #tol = s.dualTolerance
        tol = 0
        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree) &
                                     self.notBanned)[0]


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
            ret = indicesToConsider[ind]
            del indicesToConsider  # not sure if this is necessary
            del rc2  # HUGE memory leak otherwise
            return ret
        return -1
        return self.pivotColumnFirst()

    def saveWeights(self, model, mode):
        self.clpModel = model

    def isPivotAcceptable(self):
#        #TODO ComplementarityList can be defined in the current class
        s = self.clpModel
        #cl = s.getComplementarityList()
        cl = self.complementarityList
        pivotRow = s.pivotRow()
        if pivotRow < 0:
            colInd = s.sequenceIn()
            return 1

        pivotVariable = s.getPivotVariable()
        leavingVarIndex = pivotVariable[pivotRow]
        colInd = s.sequenceIn()

        if s.CLP_getVarStatus(cl[colInd]) == 1 and \
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

        del self.notBanned
        self.notBanned = self.orgBan.copy()

        return 1

    def setComplement(self, model, v1, v2):
        v1n = v1.name
        v2n = v2.name
        listv1 = np.array(model.inds.varIndex[v1n])[v1.indices]
        listv2 = np.array(model.inds.varIndex[v2n])[v2.indices]
        for i, j in izip(listv1, listv2):
            self.complementarityList[i], self.complementarityList[j] = j, i

