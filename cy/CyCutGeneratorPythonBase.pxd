from CyCglCutGeneratorBase cimport *
cimport cython.operator.dereference as deref
from CyLP.cy.CyClpSimplex cimport CyClpSimplex, CppIClpSimplex
from CyLP.cy.CyCoinIndexedVector cimport CyCoinIndexedVector

cdef class CyCglCutGeneratorBase(CyCglCutGeneratorBase):
	cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)
	cdef CyCglCutGenerator * clone(self)
	cdef object cutGeneratorObject
