# cython: embedsignature=True


cdef class CyWolfePivot(CyClpPrimalColumnPivotBase):

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

        m = self.cyModel

        # Update the reduced costs, for both the original and the slack variables

        s = self.cyModel

        if updates.nElements:
            m.updateColumnTranspose(spareRow2, updates)
            m.transposeTimes(-1, updates, spareCol2, spareCol1)
            m.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            m.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

        # cdef CyClpSimplex.Status s
        cdef double* reducedCosts = self.getReducedCosts()
        cdef int dim = self.nCols() + self.nRows()

        cdef double bestDj = self.cyModel.dualTolerance
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
