
cdef extern from "CglTreeInfo.hpp":
    cdef cppclass CppCglTreeInfo "CglTreeInfo":
        pass
    CppCglTreeInfo *new_CppCglTreeInfo "new CglTreeInfo" ()

cdef class CyCglTreeInfo:
    cdef CppCglTreeInfo* CppSelf
    cdef setCppSelf(self, CppCglTreeInfo* s)
