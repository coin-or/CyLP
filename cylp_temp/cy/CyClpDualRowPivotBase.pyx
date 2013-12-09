# cython: profile=True
# cython: embedsignature=True

np.import_array()

cimport cylp.cy.CyClpDualRowPivotBase
#from cylp.cy import CyClpSimplex

cdef int RunPivotRow(void *ptr):
    return (<CyClpDualRowPivotBase>(ptr)).pivotRow()

cdef CyClpDualRowPivot* RunDualPivotClone(void *ptr, bint copyData):
    return (<CyClpDualRowPivotBase>(ptr)).clone(copyData)

cdef double RunUpdateWeights(void *ptr,CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn):
    return (<CyClpDualRowPivotBase>(ptr)).updateWeights(inp, spare, spare2, updatedColumn)

cdef void RunUpdatePrimalSolution(void *ptr,CppCoinIndexedVector * inp,
                                       double theta,
                                       double * changeInObjective):
    cdef np.npy_intp shape = <np.npy_intp> 1
    cho = np.PyArray_SimpleNewFromData(1, &shape, np.NPY_DOUBLE, <void*>changeInObjective)
    (<CyClpDualRowPivotBase>(ptr)).updatePrimalSolution(inp, theta, cho)


cdef class CyClpDualRowPivotBase:
    def __init__(self):
        self.CppSelf = new CppClpDualRowPivotBase(
            <cpy_ref.PyObject*>self,
            <runPivotRow_t>RunPivotRow,
            <runDualPivotClone_t>RunDualPivotClone,
            <runUpdateWeights_t>RunUpdateWeights,
            <runUpdatePrimalSolution_t>RunUpdatePrimalSolution)

    cdef pivotRow(self):
        raise Exception('CyClpDualRowPivotBase.pyx: pivotRow() should' \
                        ' be implemented.')

    cdef CyClpDualRowPivot* clone(self, bint copyData):
        cdef CyClpDualRowPivot* ret =  \
                <CyClpDualRowPivot*> new CppClpDualRowPivotBase(
                                            <cpy_ref.PyObject*>self,
                                            <runPivotRow_t>RunPivotRow,
                                            <runDualPivotClone_t>RunDualPivotClone,
                                            <runUpdateWeights_t>RunUpdateWeights,
                                            <runUpdatePrimalSolution_t>RunUpdatePrimalSolution)
        return ret

    cdef double updateWeights(self, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn):
        raise Exception('CyClpDualRowPivotBase.pyx: updateWeights should ' \
                        'be implemented.')

    cdef void updatePrimalSolution(self, CppCoinIndexedVector * inp,
                                       double theta,
                                       np.ndarray[np.double_t,ndim=1] changeInObjective):
        raise Exception('CyClpDualRowPivotBase.pyx: updatePrimalSolution should ' \
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
