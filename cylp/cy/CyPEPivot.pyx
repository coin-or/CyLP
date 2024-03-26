# cython: embedsignature=True

import numpy as np
cimport numpy as np
from cylp.cy cimport CyPEPivot
#from CyClpSimplex cimport CyClpSimplex
#cimport cython.operator.dereference as deref

cdef class CyPEPivot(CyClpPrimalColumnPivotBase):
    def __init__(self, cyModel):
        self.cyModel = cyModel
        CyClpPrimalColumnPivotBase.__init__(self)

    cdef pivotColumn(self, CppCoinIndexedVector* cppupdates, CppCoinIndexedVector* cppspareRow1,
                    CppCoinIndexedVector* cppspareRow2, CppCoinIndexedVector* cppspareCol1,
                    CppCoinIndexedVector* cppspareCol2):
        updates = CyCoinIndexedVector()
        updates.setCppSelf(cppupdates)
        spareRow1 = CyCoinIndexedVector()
        spareRow1.setCppSelf(cppspareRow1)
        spareRow2 = CyCoinIndexedVector()
        spareRow2.setCppSelf(cppspareRow2)
        spareCol1 = CyCoinIndexedVector()
        spareCol1.setCppSelf(cppspareCol1)
        spareCol2 = CyCoinIndexedVector()
        spareCol2.setCppSelf(cppspareCol2)

        s = self.cyModel

        # Update the reduced costs, for both the original and the slack variables
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        rc = s.reducedCosts

        cdef double tol = s.dualTolerance

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
            print(s.iteration, ' : comp selected', maxCompInd)
            return maxCompInd

        print(s.iteration, ' : incomp selected', maxInd)
        self.updateP()
        self.updateW()
        #print('updated')
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
