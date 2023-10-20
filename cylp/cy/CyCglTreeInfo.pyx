cdef class CyCglTreeInfo:
    'CyCglTreeInfo documentation'
    def __cinit__(self):
        self.CppSelf = new CppCglTreeInfo()

    cdef setCppSelf(self, CppCglTreeInfo* s):
        del self.CppSelf
        self.CppSelf = s

