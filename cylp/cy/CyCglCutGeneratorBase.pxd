from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from cpython.ref cimport PyObject
from cylp.cy cimport CyClpSimplex
from cylp.cy.CyOsiCuts cimport CppOsiCuts
from cylp.cy.CyOsiSolverInterface cimport CppOsiSolverInterface
from cylp.cy.CyCglTreeInfo cimport CppCglTreeInfo
from cpython cimport Py_INCREF, Py_DECREF

#import numpy as np
#cimport numpy as np

cdef extern from "CglCutGenerator.hpp":
    cdef cppclass CppCglCutGenerator "CglCutGenerator":
        pass
    #CppCglCutGenerator* new_CyCglCutGenerator \
    #                                "new ICglCutGenerator" ()

cdef extern from "ICglCutGeneratorBase.h":
    #cdef cppclass CoinIndexedVector:
    #   pass
    ctypedef int (*runGenerateCuts_t)(void *obj,
                                     CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)

    ctypedef CppCglCutGenerator* (*runCglClone_t)(void *obj)

    cdef cppclass CppCglCutGeneratorBase:

        CppCglCutGeneratorBase(PyObject* obj,
                                    runGenerateCuts_t runGenerateCuts,
                                    runCglClone_t runCglClone)

        void generateCuts(CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)

        CppCglCutGenerator* clone()


cdef void RunGenerateCuts(void *ptr, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)

cdef CppCglCutGenerator* RunCglClone(void *ptr)

cdef class CyCglCutGeneratorBase:
    cdef CppCglCutGeneratorBase* CppSelf
    cdef CyClpSimplex.CyClpSimplex cyModel
    #cpdef cyModel
    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info)
    cdef CppCglCutGenerator * clone(self)

