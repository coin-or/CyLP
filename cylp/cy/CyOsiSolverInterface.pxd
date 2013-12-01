from cylp.cy.CyClpSimplex cimport CyClpSimplex, CppIClpSimplex

cdef extern from "ClpSimplex.hpp":
    cdef cppclass CppClpSimplex "ClpSimplex":
        pass

cdef extern from "OsiClpSolverInterface.hpp":
    cdef cppclass CppOsiClpSolverInterface "OsiClpSolverInterface":
        CppClpSimplex * getModelPtr()

cdef extern from "OsiSolverInterface.hpp":
    cdef cppclass CppOsiSolverInterface "OsiSolverInterface":
        pass

cdef class CyOsiSolverInterface:
    cdef CppOsiSolverInterface* CppSelf
    cdef setCppSelf(self, CppOsiSolverInterface* s)
