from cylp.cy.CyCglCutGeneratorBase cimport *
from cylp.cy.CyOsiSolverInterface cimport CppOsiSolverInterface, CyOsiSolverInterface
from cylp.cy.CyOsiCuts cimport CppOsiCuts, CyOsiCuts
from cylp.cy.CyCglTreeInfo cimport CppCglTreeInfo, CyCglTreeInfo
cimport cython.operator.dereference as deref
from cylp.cy.CyClpSimplex cimport CyClpSimplex, CppIClpSimplex
from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector
from cpython cimport Py_INCREF, Py_DECREF

cdef class CyCutGeneratorPythonBase(CyCglCutGeneratorBase):
    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)
    cdef CppCglCutGenerator * clone(self)
    cdef object cutGeneratorObject
    cdef object cyLPModel
