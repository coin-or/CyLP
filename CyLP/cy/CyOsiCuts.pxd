cimport numpy as np
import numpy as np


cdef extern from "IOsiCuts.hpp":
    cdef cppclass CppOsiCuts "CppOsiCuts":
        void addColumnCut(int size, int* lowerBoundInds, double* lowerBoundElements,
                        int* upperBoundInds, double* upperBoundElements)
        void addRowCut(int size, int* indices, double* elements,
                   double lowerBound, double upperBound)
        void printCuts()
        int sizeRowCuts()
        int sizeColCuts()
        int sizeCuts()

    CppOsiCuts *new_CppOsiCuts "new CppOsiCuts" ()


cdef class CyOsiCuts:
    cdef CppOsiCuts* CppSelf
    cdef setCppSelf(self, CppOsiCuts* s)
