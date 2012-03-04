from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from cpython.ref cimport PyObject
from CyLP.cy cimport CyClpSimplex
from CyLP.cy.CyCoinIndexedVector cimport CppCoinIndexedVector
#import numpy as np
#cimport numpy as np

cdef extern from "ClpPrimalColumnPivot.hpp":
    cdef cppclass CyClpPrimalColumnPivot "ClpPrimalColumnPivot":
        pass
    CyClpPrimalColumnPivot* new_CyClpPrimalColumnPivot \
                                    "new ClpPrimalColumnPivot" ()

cdef extern from "ClpFactorization.hpp":
    cdef cppclass ClpFactorization:
        int updateColumnTranspose (CppCoinIndexedVector * regionSparse,
                  CppCoinIndexedVector * regionSparse2)

cdef extern from "IClpPrimalColumnPivotBase.h":
    #cdef cppclass CoinIndexedVector:
    #   pass
    ctypedef int (*runPivotColumn_t)(void *obj,
                                     CppCoinIndexedVector*,
                                     CppCoinIndexedVector*,
                                     CppCoinIndexedVector*,
                                     CppCoinIndexedVector*,
                                     CppCoinIndexedVector*)
    ctypedef CyClpPrimalColumnPivot* (*runClone_t)(void *obj, bint copyData)
    ctypedef void (*runSaveWeights_t)(void* obj,
                                      CyClpSimplex.CppIClpSimplex* model,
                                      int mode)
    cdef cppclass CppClpPrimalColumnPivotBase:
        CyClpSimplex.CppIClpSimplex* model()
        void setModel(CyClpSimplex.CppIClpSimplex* m)
        CppClpPrimalColumnPivotBase(PyObject* obj,
                                    runPivotColumn_t runPivotColumn,
                                    runClone_t runClone,
                                    runSaveWeights_t runSaveWeights)

        int pivotColumn(CppCoinIndexedVector*,
                        CppCoinIndexedVector*,
                        CppCoinIndexedVector*,
                        CppCoinIndexedVector*,
                        CppCoinIndexedVector*)

        int DantzigDualUpdate(CppCoinIndexedVector * updates,
                CppCoinIndexedVector * spareRow1,
              CppCoinIndexedVector * spareRow2,
              CppCoinIndexedVector * spareColumn1,
              CppCoinIndexedVector * spareColumn2)

        CyClpPrimalColumnPivot* clone(bint copyData)
        void saveWeights(CyClpSimplex.CppIClpSimplex * model,
                         int mode)


cdef int RunPivotColumn(void *ptr,
                        CppCoinIndexedVector* v1,
                        CppCoinIndexedVector* v2,
                        CppCoinIndexedVector* v3,
                        CppCoinIndexedVector* v4,
                        CppCoinIndexedVector* v5)

cdef CyClpPrimalColumnPivot* RunClone(void *ptr, bint copyData)
cdef void RunSaveWeights(void *ptr, CyClpSimplex.CppIClpSimplex * model,
                         int mode)


cdef class CyClpPrimalColumnPivotBase:
    cdef CppClpPrimalColumnPivotBase* CppSelf
    cdef CyClpSimplex.CyClpSimplex cyModel
    #cpdef cyModel
    cdef pivotColumn(self, CppCoinIndexedVector* v1,
                        CppCoinIndexedVector* v2,
                        CppCoinIndexedVector* v3,
                        CppCoinIndexedVector* v4,
                        CppCoinIndexedVector* v5)

    cdef int DantzigDualUpdate(self, CppCoinIndexedVector * updates,
                CppCoinIndexedVector * spareRow1,
              CppCoinIndexedVector * spareRow2,
              CppCoinIndexedVector * spareColumn1,
              CppCoinIndexedVector * spareColumn2)

    cdef CyClpPrimalColumnPivot * clone(self, bint copyData)
    cdef void saveWeights(self, CyClpSimplex.CppIClpSimplex * model,
                          int mode)
    cdef CyClpSimplex.CppIClpSimplex* model(self)
    cdef void setModel(self, CyClpSimplex.CppIClpSimplex* m)
    cdef double* getReducedCosts(self)
