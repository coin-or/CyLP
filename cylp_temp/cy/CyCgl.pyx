#from cylp.cy cimport CyCgl


cdef class CyCglCutGenerator:
    pass

cdef class CyCglAllDifferent:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglAllDifferent()

cdef class CyCglClique:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglClique()

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

cdef class CyCglOddHole:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglOddHole()

#######################

cdef class CyCglFlowCover:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglFlowCover()

######################

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

cdef class CyCglRedSplit:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglRedSplit()

##############################

cdef class CyCglLiftAndProject:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglLiftAndProject()

cdef class CyCglLandP:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglLandP()

#################################

cdef class CyCglMixedIntegerRounding:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglMixedIntegerRounding()

cdef class CyCglMixedIntegerRounding2:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglMixedIntegerRounding2()

cdef class CyCglTwomir:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglTwomir()


cdef class CyCglResidualCapacity:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglResidualCapacity()


###########################


#cdef class CyCglDuplicateRow:
#    def __cinit__(self):
#        self.CppSelf = <CppCglCutGenerator*>new CppCglDuplicateRow()

cdef class CyCglPreProcess:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglPreProcess()

cdef class CyCglProbing:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglProbing()

cdef class CyCglSimpleRounding:
    def __cinit__(self):
        self.CppSelf = <CppCglCutGenerator*>new CppCglSimpleRounding()


