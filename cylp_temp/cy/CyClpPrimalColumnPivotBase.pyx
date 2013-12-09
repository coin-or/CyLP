# cython: profile=True
# cython: embedsignature=True

cimport cylp.cy.CyClpPrimalColumnPivotBase
#from cylp.cy import CyClpSimplex

cdef int RunPivotColumn(void *ptr, CppCoinIndexedVector* v1,
                        CppCoinIndexedVector* v2,
                        CppCoinIndexedVector* v3,
                        CppCoinIndexedVector* v4,
                        CppCoinIndexedVector* v5):
    return (<CyClpPrimalColumnPivotBase>(ptr)).pivotColumn(v1, v2, v3, v4, v5)

cdef CyClpPrimalColumnPivot* RunClone(void *ptr, bint copyData):
    return (<CyClpPrimalColumnPivotBase>(ptr)).clone(copyData)

cdef void RunSaveWeights(void *ptr,
                         CyClpSimplex.CppIClpSimplex * model,
                         int mode):
    (<CyClpPrimalColumnPivotBase>(ptr)).saveWeights(model, mode)


cdef class CyClpPrimalColumnPivotBase:
    def __init__(self):
        self.CppSelf = new CppClpPrimalColumnPivotBase(
            <cpy_ref.PyObject*>self,
            <runPivotColumn_t>RunPivotColumn,
            <runClone_t>RunClone,
            <runSaveWeights_t>RunSaveWeights)
        Py_INCREF(self)

    def __dealloc__(self):
        Py_DECREF(self)
        del self.CppSelf

    cdef pivotColumn(self, CppCoinIndexedVector* v1, CppCoinIndexedVector* v2,
                        CppCoinIndexedVector* v3, CppCoinIndexedVector* v4,
                        CppCoinIndexedVector* v5):
        raise Exception('CyClpPrimalColumnPivotBase.pyx: pivotColumn must' \
                        ' be implemented.')

    cdef CyClpPrimalColumnPivot* clone(self, bint copyData):
        cdef CyClpPrimalColumnPivot* ret =  \
                <CyClpPrimalColumnPivot*> new CppClpPrimalColumnPivotBase(
                                            <cpy_ref.PyObject*>self,
                                            <runPivotColumn_t>RunPivotColumn,
                                            <runClone_t>RunClone,
                                            <runSaveWeights_t>RunSaveWeights)
        return ret

    cdef void saveWeights(self, CyClpSimplex.CppIClpSimplex * model, int mode):
        raise Exception('CyClpPrimalColumnPivotBase.pyx: saveWeights must ' \
                        'be implemented.')

    cdef CyClpSimplex.CppIClpSimplex* model(self):
        return self.CppSelf.model()

    cdef void setModel(self, CyClpSimplex.CppIClpSimplex* m):
        self.CppSelf.setModel(m)

    cdef double* getReducedCosts(self):
        return self.model().djRegion()

    property nRows:
        def __get__(self):
            return self.CppSelf.model().getNumRows()

    property nCols:
        def __get__(self):
            return self.CppSelf.model().getNumCols()
