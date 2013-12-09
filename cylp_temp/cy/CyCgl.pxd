cimport numpy as np
from cpython.ref cimport PyObject

#############################
#Combinatorial cuts:
#############################
cdef extern from "CglAllDifferent.hpp":
    cdef cppclass CppCglAllDifferent "CglAllDifferent":
        pass

cdef extern from "CglClique.hpp":
    cdef cppclass CppCglClique "CglClique":
        pass

cdef extern from "CglKnapsackCover.hpp":
    cdef cppclass CppCglKnapsackCover "CglKnapsackCover":
        int getMaxInKnapsack()
        void setMaxInKnapsack(int)

cdef extern from "CglOddHole.hpp":
    cdef cppclass CppCglOddHole "CglOddHole":
        pass

#############################
#Flow cover cuts:
#############################

cdef extern from "CglFlowCover.hpp":
    cdef cppclass CppCglFlowCover "CglFlowCover":
        pass

#############################
#Gomory cuts and variants:
#############################

cdef extern from "CglGomory.hpp":
    cdef cppclass CppCglGomory "CglGomory":
        int getLimit()
        void setLimit(int)

cdef extern from "CglRedSplit.hpp":
    cdef cppclass CppCglRedSplit "CglRedSplit":
        pass

#############################
#Lift-and-project cuts:
#############################

cdef extern from "CglLiftAndProject.hpp":
    cdef cppclass CppCglLiftAndProject "CglLiftAndProject":
        pass

cdef extern from "CglLandP.hpp":
    cdef cppclass CppCglLandP "CglLandP":
        pass

#############################
#Mixed integer rounding cuts and variants:
#############################

cdef extern from "CglMixedIntegerRounding.hpp":
    cdef cppclass CppCglMixedIntegerRounding "CglMixedIntegerRounding":
        pass

cdef extern from "CglMixedIntegerRounding2.hpp":
    cdef cppclass CppCglMixedIntegerRounding2 "CglMixedIntegerRounding2":
        pass

cdef extern from "CglTwomir.hpp":
    cdef cppclass CppCglTwomir "CglTwomir":
        pass

cdef extern from "CglResidualCapacity.hpp":
    cdef cppclass CppCglResidualCapacity "CglResidualCapacity":
        pass

#############################
#Strengthening:
#############################

#cdef extern from "CglDuplicateRow.hpp":
#    cdef cppclass CppCglDuplicateRow "CglDuplicateRow":
#        pass

cdef extern from "CglPreProcess.hpp":
    cdef cppclass CppCglPreProcess "CglPreProcess":
        pass

cdef extern from "CglProbing.hpp":
    cdef cppclass CppCglProbing "CglProbing":
        pass

cdef extern from "CglSimpleRounding.hpp":
    cdef cppclass CppCglSimpleRounding "CglSimpleRounding":
        pass

## parent of all cuts
cdef extern from "CglCutGenerator.hpp":
    cdef cppclass CppCglCutGenerator "CglCutGenerator":
        pass


cdef class CyCglCutGenerator:
    cdef CppCglCutGenerator* CppSelf

###########

cdef class CyCglAllDifferent(CyCglCutGenerator):
    pass

cdef class CyCglClique(CyCglCutGenerator):
    pass

cdef class CyCglKnapsackCover(CyCglCutGenerator):
    cdef CppCglKnapsackCover* realCppSelf(self)

cdef class CyCglOddHole(CyCglCutGenerator):
    pass

##################

cdef class CyCglFlowCover(CyCglCutGenerator):
    pass

##################

cdef class CyCglGomory(CyCglCutGenerator):
    cdef CppCglGomory* realCppSelf(self)
    #cdef CppCglGomory* CppSelf
    #cdef CppCglCutGenerator* CppSelf

cdef class CyCglRedSplit(CyCglCutGenerator):
    pass

###################

cdef class CyCglLiftAndProject(CyCglCutGenerator):
    pass

cdef class CyCglLandP(CyCglCutGenerator):
    pass

###################

cdef class CyCglMixedIntegerRounding(CyCglCutGenerator):
    pass

cdef class CyCglMixedIntegerRounding2(CyCglCutGenerator):
    pass

cdef class CyCglTwomir(CyCglCutGenerator):
    pass

cdef class CyCglResidualCapacity(CyCglCutGenerator):
    pass

####################

#cdef class CyCglDuplicateRow(CyCglCutGenerator):
#    pass

cdef class CyCglPreProcess(CyCglCutGenerator):
    pass

cdef class CyCglProbing(CyCglCutGenerator):
    pass

cdef class CyCglSimpleRounding(CyCglCutGenerator):
    pass

