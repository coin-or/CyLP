
cimport CyOsiCuts

cdef class CyOsiCuts:
    'CyOsiCuts documentation'
    def __cinit__(self):
        self.CppSelf = new CppOsiCuts()

    cdef setCppSelf(self, CppOsiCuts* s):
        del self.CppSelf
        self.CppSelf = s

