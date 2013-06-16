from CyClpPrimalColumnPivotBase cimport *
cimport cython.operator.dereference as deref
from CyLP.cy.CyClpSimplex cimport CyClpSimplex, CppIClpSimplex
from CyLP.cy.CyCoinIndexedVector cimport CyCoinIndexedVector

cdef class CyPivotPythonBase(CyClpPrimalColumnPivotBase):
	cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
					 CppCoinIndexedVector* v4, CppCoinIndexedVector* v5)
	cdef CyClpPrimalColumnPivot * clone(self, bint copyData)
	cdef void saveWeights(self, CppIClpSimplex * model, int mode)
	cdef object pivotMethodObject
