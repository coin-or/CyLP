
cdef extern from "OsiCuts.hpp":
    cdef cppclass CppOsiCuts "OsiCuts":
        pass
    CppOsiCuts *new_CppOsiCuts "new OsiCuts" ()

cdef class CyOsiCuts:
    cdef CppOsiCuts* CppSelf
    cdef setCppSelf(self, CppOsiCuts* s)
