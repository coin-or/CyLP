
cdef extern from "OsiSolverInterface.hpp":
    cdef cppclass CppOsiSolverInterface "OsiSolverInterface":
        pass

cdef class CyOsiSolverInterface:
    cdef CppOsiSolverInterface* CppSelf
    cdef setCppSelf(self, CppOsiSolverInterface* s)
