from CyClpDualRowPivotBase cimport *
cimport cython.operator.dereference as deref
from CyLP.cy cimport CyClpSimplex
from CyLP.cy.CyCoinIndexedVector cimport CyCoinIndexedVector


cdef class CyDualPivotPythonBase(CyClpDualRowPivotBase):
    cdef pivotRow(self)
    cdef CyClpDualRowPivot * clone(self, bint copyData)
    cdef double updateWeights(self, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)

    cdef void updatePrimalSolution(self, CppCoinIndexedVector* inp,
                                         double theta,
                                         double * changeInObjective)

    cdef object dualPivotMethodObject
