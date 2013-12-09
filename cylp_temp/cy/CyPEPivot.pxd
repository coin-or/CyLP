cimport cython.operator.dereference as deref
from CyClpPrimalColumnPivotBase cimport *
from cylp.cy cimport CyClpSimplex
from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector


cdef class CyPEPivot(CyClpPrimalColumnPivotBase):
    cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
                     CppCoinIndexedVector* v4, CppCoinIndexedVector* v5)
    cdef CyClpPrimalColumnPivot * clone(self, bint copyData)
    cdef void saveWeights(self, CyClpSimplex.CppIClpSimplex* model, int mode)
