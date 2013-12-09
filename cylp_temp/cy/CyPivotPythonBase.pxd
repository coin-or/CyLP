from CyClpPrimalColumnPivotBase cimport *
cimport cython.operator.dereference as deref
from cylp.cy.CyClpSimplex cimport CyClpSimplex, CppIClpSimplex
from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector
from cpython cimport Py_INCREF, Py_DECREF

cdef class CyPivotPythonBase(CyClpPrimalColumnPivotBase):
	cdef pivotColumn(self, CppCoinIndexedVector* v1,
                     CppCoinIndexedVector* v2, CppCoinIndexedVector* v3,
					 CppCoinIndexedVector* v4, CppCoinIndexedVector* v5)
	cdef CyClpPrimalColumnPivot * clone(self, bint copyData)
	cdef void saveWeights(self, CppIClpSimplex * model, int mode)
	cdef object pivotMethodObject
