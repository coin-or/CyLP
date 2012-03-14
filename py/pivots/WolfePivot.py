import numpy as np
from PivotPythonBase import PivotPythonBase


class WolfePivot(PivotPythonBase):

    def __init__(self, clpModel, bucketSize=1):
        self.dim = clpModel.nConstraints + clpModel.nVariables
        self.clpModel = clpModel

    def pivotColumn(self):
        s = self.clpModel
        
        # If objective function linear proceed as normal 
        if s.Hessian == None:
            rc = s.reducedCosts
        else:
            x = s.primalVariableSolutionAll
            G = s.Hessian
            
            #print '1'
            dim = s.nVariables + s.nConstraints
            #print dim
            G[dim - 1, dim - 1] = 0 
            #print 'x shape = ', x.shape, x.__class__
            #print 'G shape = ', G.shape, G.__class__
            #print G * x
            #print s.reducedCosts
            rc = G * x + s.reducedCosts
            #print '2'

        tol = s.dualTolerance()
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

        rc2 = abs(rc[indicesToConsider])

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
            return  indicesToConsider[ind]
        return -1
        return self.pivotColumnFirst()


    def isPivotAcceptable(self):
        return True
#        #TODO ComplementarityList can be defined in the current class
#        s = self.clpModel
#        cl = s.getComplementarityList()
#        pivotRow = s.pivotRow()
#        if pivotRow < 0:
#            return 1
#
#        pivotVariable = s.getPivotVariable()
#        leavingVarIndex = pivotVariable[pivotRow]
#        colInd = s.sequenceIn()
#
#        if s.getStatus(cl[colInd]) == 1 and \
#            cl[colInd] != leavingVarIndex:
#            #print colInd , ' flagged'
#            #self.banList[colInd] = 1
#            self.banList.append(colInd)
#
#            #s.setFlagged(colInd)
#            return 0
#
#        #self.banList = np.zeros(self.dim, np.int)
#        self.banList = []
#
#        return 1
