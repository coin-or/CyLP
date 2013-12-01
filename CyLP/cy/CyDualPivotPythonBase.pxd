import numpy as np
cimport numpy as np
from CyClpDualRowPivotBase cimport *
cimport cython.operator.dereference as deref
from cylp.cy cimport CyClpSimplex
from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector
from cpython cimport Py_INCREF, Py_DECREF

cdef class CyDualPivotPythonBase(CyClpDualRowPivotBase):
    cdef pivotRow(self)
    cdef CyClpDualRowPivot * clone(self, bint copyData)
    cdef double updateWeights(self, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn)

    cdef void updatePrimalSolution(self, CppCoinIndexedVector* inp,
                                         double theta,
                                         np.ndarray[np.double_t,ndim=1] changeInObjective)

    cdef object dualPivotMethodObject
