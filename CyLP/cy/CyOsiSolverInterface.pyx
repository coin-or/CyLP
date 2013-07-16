
cimport CyOsiSolverInterface

cdef class CyOsiSolverInterface:
    'CyOsiSolverInterface documentation'
    def __cinit__(self):
        pass

    cdef setCppSelf(self, CppOsiSolverInterface* s):
        del self.CppSelf
        self.CppSelf = s

