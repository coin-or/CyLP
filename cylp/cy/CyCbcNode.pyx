# cython: embedsignature=True

cdef class CyCbcNode:
    def __cinit__(self):
        self.CppSelf = new CppICbcNode()

    cdef setCppSelf(self, CppICbcNode* cbcnode):
        self.CppSelf = cbcnode
        return self

    property depth:
        def __get__(self):
            return self.CppSelf.depth()

    property numberUnsatisfied:
        def __get__(self):
            return self.CppSelf.numberUnsatisfied()

    property sumInfeasibilities:
        def __get__(self):
            return self.CppSelf.sumInfeasibilities()

    property active:
        def __get__(self):
            return self.CppSelf.active()

    property onTree:
        def __get__(self):
            return self.CppSelf.onTree()

    property objectiveValue:
        def __get__(self):
            return self.CppSelf.objectiveValue()

    def breakTie(self, CyCbcNode y):
        return self.CppSelf.breakTie(y.CppSelf)
