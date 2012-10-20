#from CyLP.cy cimport CyCgl


cdef class CyCglCutGenerator:
    pass

cdef class CyCglProbing:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglProbing()

cdef class CyCglGomory(CyCglCutGenerator):
    def __cinit__(self, limit=50):
        self.CppSelf = <CppCglCutGenerator*>new CppCglGomory()
        self.limit = limit

    cdef CppCglGomory* realCppSelf(self):
        return <CppCglGomory*>self.CppSelf

    property limit:
        def __get__(self):
            return self.realCppSelf().getLimit()

        def __set__(self, limit):
            self.realCppSelf().setLimit(limit)


cdef class CyCglClique:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglClique()

cdef class CyCglFlowCover:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglFlowCover()

cdef class CyCglKnapsackCover:
    def __cinit__(self, maxInKnapsack=50):
        self.CppSelf = <CppCglCutGenerator*>new CppCglKnapsackCover()
        self.maxInKnapsack = maxInKnapsack

    cdef CppCglKnapsackCover* realCppSelf(self):
        return <CppCglKnapsackCover*>self.CppSelf

    property maxInKnapsack:
        def __get__(self):
            return self.realCppSelf().getMaxInKnapsack()

        def __set__(self, limit):
            self.realCppSelf().setMaxInKnapsack(limit)


cdef class CyCglMixedIntegerRounding2:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglMixedIntegerRounding2()

cdef class CyCglRedSplit:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglRedSplit()
