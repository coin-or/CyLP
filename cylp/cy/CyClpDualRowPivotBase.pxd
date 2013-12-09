from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from cpython.ref cimport PyObject
from cylp.cy cimport CyClpSimplex
from cylp.cy.CyCoinIndexedVector cimport CppCoinIndexedVector
import numpy as np
cimport numpy as np

cdef extern from "ClpDualRowPivot.hpp":
    cdef cppclass CyClpDualRowPivot "ClpDualRowPivot":
        pass
    CyClpDualRowPivot* new_CyClpDualRowPivot \
                                    "new ClpDualRowPivot" ()

cdef extern from "ClpFactorization.hpp":
    cdef cppclass ClpFactorization:
        int updateColumnTranspose (CppCoinIndexedVector * regionSparse,
                  CppCoinIndexedVector * regionSparse2)

cdef extern from "IClpDualRowPivotBase.h":
    #cdef cppclass CoinIndexedVector:
    #   pass
    ctypedef int (*runPivotRow_t)(void *obj)
    ctypedef CyClpDualRowPivot* (*runDualPivotClone_t)(void *obj, bint copyData)
    ctypedef double (*runUpdateWeights_t)(void* obj,
                                  CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)

    ctypedef void (*runUpdatePrimalSolution_t)(void* obj,
                                       CppCoinIndexedVector * inp,
                                       double theta,
                                       double * changeInObjective)

    cdef cppclass CppClpDualRowPivotBase:
        CyClpSimplex.CppIClpSimplex* model()
        void setModel(CyClpSimplex.CppIClpSimplex* m)
        CppClpDualRowPivotBase(PyObject* obj,
                                    runPivotRow_t runPivotRow,
                                    runDualPivotClone_t runDualPivotClone,
                                    runUpdateWeights_t runUpdateWeights,
                                    runUpdatePrimalSolution_t runUpdatePrimalSolution)

        int pivotRow()

        CyClpDualRowPivot* clone(bint copyData)
        double updateWeights(CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)

        void updatePrimalSolution(CppCoinIndexedVector * inp,
                                       double theta,
                                       double * changeInObjective)



cdef int RunPivotRow(void *ptr)

cdef CyClpDualRowPivot* RunDualPivotClone(void *ptr, bint copyData)
cdef double RunUpdateWeights(void *ptr, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)
cdef void RunUpdatePrimalSolution(void *ptr,
                                       CppCoinIndexedVector * inp,
                                       double theta,
                                       double * changeInObjective)


cdef class CyClpDualRowPivotBase:
    cdef CppClpDualRowPivotBase* CppSelf
    cdef CyClpSimplex.CyClpSimplex cyModel
    #cpdef cyModel
    cdef pivotRow(self)

    cdef CyClpDualRowPivot * clone(self, bint copyData)
    cdef double updateWeights(self, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)
    cdef void updatePrimalSolution(self, CppCoinIndexedVector * inp,
                                       double theta,
                                       np.ndarray[np.double_t,ndim=1] changeInObjective)
    cdef CyClpSimplex.CppIClpSimplex* model(self)
    cdef void setModel(self, CyClpSimplex.CppIClpSimplex* m)
    cdef double* getReducedCosts(self)
