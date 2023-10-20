# cython: embedsignature=True


import numpy as np
cimport numpy as np

# varStatus defined to mimic 'enum ClpSimplex::Status'.
# But is too slow obviously
varStatus = ['isFree', 'basic', 'atUpperBound',
             'atLowerBound', 'superBasic', 'isFixed']

cdef class CyDantzigPivot(CyClpPrimalColumnPivotBase):
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

        #self.DantzigDualUpdate(v1, v2, v3, v4, v5)
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

        #incides of vars not fixed and not flagged
        #indicesToConsider = np.where((status & 7 != 1) & (status & 7 != 5) &
        #        (status & 64 == 0) & (((rc > tol) & (status & 7 == 2)) |
        #            ((rc < -tol) & (status & 7 == 3))) )[0]

        cdef np.ndarray indicesToConsider = \
                                np.where(s.varNotFlagged & s.varNotFixed &
                                     s.varNotBasic &
                                     (((rc > tol) & s.varIsAtUpperBound) |
                                     ((rc < -tol) & s.varIsAtLowerBound) |
                                     s.varIsFree))[0]

        #freeVarInds = np.where(s.varIsFree)
        #rc[freeVarInds] *= 10

        cdef np.ndarray rc2 = abs(rc[indicesToConsider])

        cdef int checkFree = 0
        cdef int ind
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
