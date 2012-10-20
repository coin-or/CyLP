cimport numpy as np
from cpython.ref cimport PyObject

cdef extern from "CglCutGenerator.hpp":
    cdef cppclass CppCglCutGenerator "CglCutGenerator":
        pass

cdef extern from "CglProbing.hpp":
    cdef cppclass CppCglProbing "CglProbing":
        pass

cdef extern from "CglGomory.hpp":
    cdef cppclass CppCglGomory "CglGomory":
        int getLimit()
        void setLimit(int)

cdef extern from "CglClique.hpp":
    cdef cppclass CppCglClique "CglClique":
        pass

cdef extern from "CglFlowCover.hpp":
    cdef cppclass CppCglFlowCover "CglFlowCover":
        pass

cdef extern from "CglKnapsackCover.hpp":
    cdef cppclass CppCglKnapsackCover "CglKnapsackCover":
        int getMaxInKnapsack()
        void setMaxInKnapsack(int)

cdef extern from "CglMixedIntegerRounding2.hpp":
    cdef cppclass CppCglMixedIntegerRounding2 "CglMixedIntegerRounding2":
        pass

cdef extern from "CglRedSplit.hpp":
    cdef cppclass CppCglRedSplit "CglRedSplit":
        pass

cdef extern from "CglCutGenerator.hpp":
    cdef cppclass CppCglCutGenerator "CglCutGenerator":
        pass

cdef class CyCglCutGenerator:
    cdef CppCglCutGenerator* CppSelf

cdef class CyCglProbing(CyCglCutGenerator):
    pass
    #cdef CppCglProbing* CppSelf

cdef class CyCglGomory(CyCglCutGenerator):
    cdef CppCglGomory* realCppSelf(self)
    #cdef CppCglGomory* CppSelf
    #cdef CppCglCutGenerator* CppSelf

cdef class CyCglClique(CyCglCutGenerator):
    pass
    #cdef CppCglClique* CppSelf

cdef class CyCglFlowCover(CyCglCutGenerator):
    pass
    #cdef CppCglFlowCover* CppSelf

cdef class CyCglKnapsackCover(CyCglCutGenerator):
    cdef CppCglKnapsackCover* realCppSelf(self)
    #cdef CppCglKnapsackCover* CppSelf

cdef class CyCglMixedIntegerRounding2(CyCglCutGenerator):
    pass
    #cdef CppCglMixedIntegerRounding2* CppSelf

cdef class CyCglRedSplit(CyCglCutGenerator):
    pass
    #cdef CppCglRedSplit* CppSelf
