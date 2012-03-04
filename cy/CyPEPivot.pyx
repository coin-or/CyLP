# cython: embedsignature=True


import numpy as np
cimport numpy as np
from CyLP.cy cimport CyPEPivot
#from CyClpSimplex cimport CyClpSimplex
#cimport cython.operator.dereference as deref

cdef class CyPEPivot(CyClpPrimalColumnPivotBase):
    def __init__(self, cyModel):
        self.cyModel = cyModel
        CyClpPrimalColumnPivotBase.__init__(self)

    cdef pivotColumn(self, CppCoinIndexedVector* v1, CppCoinIndexedVector* v2,
                     CppCoinIndexedVector* v3, CppCoinIndexedVector* v4,
                     CppCoinIndexedVector* v5):
        self.DantzigDualUpdate(v1, v2, v3, v4, v5)

        s = self.cyModel
        rc = s.reducedCosts

        cdef double tol = s.dualTolerance()

        indicesToConsider = np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        rc2 = abs(rc[indicesToConsider])

        cdef double maxRc = -1
        cdef int maxInd = -1
        cdef double maxCompRc = -1
        cdef int maxCompInd = -1
        cdef int ind
        for i in xrange(len(rc2)):
            ind = indicesToConsider[i]
            if rc2[i] > maxCompRc:
                if self.isCompatible(ind):
                    maxCompRc = rc2[i]
                    maxCompInd = ind
                else:
                    pass
                if rc2[i] > maxRc:
                    maxRc = rc2[i]
                    maxInd = ind

        if maxCompInd != -1 and maxCompRc > 0.4 * maxRc:
            print s.iteration, ' : comp selected', maxCompInd
            return maxCompInd

        print s.iteration, ' : incomp selected', maxInd
        self.updateP()
        self.updateW()
        #print 'updated'
        return maxInd

    cdef CyClpPrimalColumnPivot* clone(self, bint copyData):
        cdef CyClpPrimalColumnPivot* ret =  \
                    <CyClpPrimalColumnPivot*> new CppClpPrimalColumnPivotBase(
                            <cpy_ref.PyObject*>self,
                            <runPivotColumn_t>RunPivotColumn,
                            <runClone_t>RunClone,
                            <runSaveWeights_t>RunSaveWeights)
        return ret
    cdef void saveWeights(self, CyClpSimplex.CppIClpSimplex * model, int mode):
        self.CppSelf.setModel(model)
