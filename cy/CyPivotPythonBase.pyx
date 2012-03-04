# cython: embedsignature=True


cimport CyPivotPythonBase

cdef class CyPivotPythonBase(CyClpPrimalColumnPivotBase):
    def __init__(self, pivotMethodObject):
        CyClpPrimalColumnPivotBase.__init__(self)
        self.pivotMethodObject = pivotMethodObject

    cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
                     CppCoinIndexedVector* v4, CppCoinIndexedVector* v5):
        self.DantzigDualUpdate(v1, v2, v3, v4, v5)
        return self.pivotMethodObject.pivotColumn()

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
