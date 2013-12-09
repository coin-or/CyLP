cimport numpy as np

cdef extern from "ICbcNode.hpp":
    cdef cppclass CppICbcNode "ICbcNode":
        int depth()
        int numberUnsatisfied()
        double sumInfeasibilities()
        bint active()
        bint onTree()
        double objectiveValue()
        bint breakTie(CppICbcNode* y)

cdef class CyCbcNode:
    cdef CppICbcNode* CppSelf
    cdef setCppSelf(self, CppICbcNode* cbcnode)
