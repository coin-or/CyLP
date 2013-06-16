from CyClpPrimalColumnPivotBase cimport *
from CyLP.cy cimport CyClpSimplex
from CyLP.cy.CyCoinIndexedVector cimport CyCoinIndexedVector


cdef class CyWolfePivot(CyClpPrimalColumnPivotBase):
    cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
                     CppCoinIndexedVector* v4, CppCoinIndexedVector* v5)
    cdef CyClpPrimalColumnPivot* clone(self, bint copyData)
    cdef void saveWeights(self, CyClpSimplex.CppIClpSimplex* model, int mode)
