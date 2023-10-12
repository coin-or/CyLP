# cython: embedsignature=True


cdef class CyPivotPythonBase(CyClpPrimalColumnPivotBase):
    def __init__(self, pivotMethodObject):
        CyClpPrimalColumnPivotBase.__init__(self)
        Py_INCREF(pivotMethodObject)
        self.pivotMethodObject = pivotMethodObject

    def __dealloc__(self):
        Py_DECREF(self.pivotMethodObject)

    cdef pivotColumn(self, CppCoinIndexedVector* updates,
                     CppCoinIndexedVector* spareRow1, CppCoinIndexedVector* spareRow2,
                     CppCoinIndexedVector* spareCol1, CppCoinIndexedVector* spareCol2):
        cyupdates = CyCoinIndexedVector()
        cyupdates.setCppSelf(updates)
        cyspareRow1 = CyCoinIndexedVector()
        cyspareRow1.setCppSelf(spareRow1)
        cyspareRow2 = CyCoinIndexedVector()
        cyspareRow2.setCppSelf(spareRow2)
        cyspareCol1 = CyCoinIndexedVector()
        cyspareCol1.setCppSelf(spareCol1)
        cyspareCol2 = CyCoinIndexedVector()
        cyspareCol2.setCppSelf(spareCol2)
        return self.pivotMethodObject.pivotColumn(cyupdates,
                                    cyspareRow1, cyspareRow2,
                                    cyspareCol1, cyspareCol2)

    cdef CyClpPrimalColumnPivot* clone(self, bint copyData):
        cdef CyClpPrimalColumnPivot* ret =  \
                <CyClpPrimalColumnPivot*> new CppClpPrimalColumnPivotBase(
                            <cpy_ref.PyObject*>self,
                            <runPivotColumn_t>RunPivotColumn,
                            <runClone_t>RunClone,
                            <runSaveWeights_t>RunSaveWeights)
        return ret

    cdef void saveWeights(self, CppIClpSimplex * model, int mode):
        cymodel = CyClpSimplex()
        cymodel.setCppSelf(model)
        self.pivotMethodObject.saveWeights(cymodel, mode)

