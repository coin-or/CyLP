# cython: embedsignature=True


from CyLP.cy cimport CyWolfePivot

cdef class CyWolfePivot(CyClpPrimalColumnPivotBase):

    cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
                     CppCoinIndexedVector* v4, CppCoinIndexedVector* v5):
        self.DantzigDualUpdate(v1, v2, v3, v4, v5)

        cdef double* reducedCosts = self.getReducedCosts()
        cdef int dim = self.nCols() + self.nRows()
        cdef CyClpSimplex.Status s

        cdef double bestDj = self.model().dualTolerance()
        cdef int bestSequence = -1
        cdef i = -1
        while i < dim - 1:
            i += 1
            if self.model().flagged(i):
                continue
            rc = reducedCosts[i]
            s = self.CppSelf.model().getStatus(i)
            if s == 1 or s == 5:
                continue
            elif s == 2:
                if rc > bestDj:
                    bestDj = rc
                    bestSequence = i
                    return bestSequence
            elif s == 3:
                if rc < -bestDj:
                    bestDj = -rc
                    bestSequence = i
                    return bestSequence
        return bestSequence

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
