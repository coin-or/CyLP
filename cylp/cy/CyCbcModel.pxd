cimport numpy as np
from cpython.ref cimport PyObject
from cylp.cy.CyCgl cimport CyCglCutGenerator, CppCglCutGenerator
from cylp.cy.CyCbcNode cimport CyCbcNode, CppICbcNode
from cylp.cy.CyOsiSolverInterface cimport CppOsiSolverInterface, CyOsiSolverInterface
from cpython cimport Py_INCREF, Py_DECREF


cdef extern from "CbcCompareUser.hpp":
    cdef cppclass CppCbcCompareUser "CbcCompareUser":
        pass
    ctypedef int (*runTest_t)(void* instance, CppICbcNode* x,
                              CppICbcNode* y) except? -1
    ctypedef bint (*runNewSolution_t)(void*instance, CppICbcModel* model,
                       double objectiveAtContinuous,
                       int numberInfeasibilitiesAtContinuous) except? -1
    ctypedef int (*runEvery1000Nodes_t)(void* instance,
                            CppICbcModel* model, int numberNodes) except? -1
    bint equalityTest(CppICbcNode* x, CppICbcNode* y)


cdef extern from "ICbcModel.hpp":
    cdef cppclass CppICbcModel "ICbcModel":
        PyObject* getPrimalVariableSolution()

        int cbcMain()
        int getSolutionCount()
        int getNumberHeuristicSolutions()
        int getNodeCount()
        double getObjValue()
        double getBestPossibleObjValue()
        int numberObjects()
        void setNodeCompare(PyObject* obj, runTest_t runTest,
                            runNewSolution_t runNewSolution,
                            runEvery1000Nodes_t runEvery1000Nodes)
        void addCutGenerator(CppCglCutGenerator* generator,
                        int howOften,
                        char* name,
                        bint normal,
                        bint atSolution,
                        bint infeasible,
                        int howOftenInSub,
                        int whatDepth,
                        int whatDepthInSub)
        void branchAndBound(int doStatistics)
        int status()
        int secondaryStatus()
        bint isProvenOptimal()
        bint isProvenInfeasible()
        bint isInitialSolveProvenPrimalInfeasible()
        bint isInitialSolveProvenDualInfeasible()
        bint isInitialSolveProvenOptimal()
        bint isInitialSolveAbandoned()

        bint setIntegerTolerance(double value)
        double getIntegerTolerance()

        bint setMaximumSeconds(double value)
        double getMaximumSeconds()

        bint setMaximumNodes(int value)
        int getMaximumNodes()

        int getNumberThreads()
        void setNumberThreads(int value)

        void setLogLevel(int value)
        int logLevel()

        void setAllowableFractionGap(double value)
        double getAllowableFractionGap()

        void setAllowablePercentageGap(double value)
        double getAllowablePercentageGap()

        void setAllowableGap(double value)
        double getAllowableGap()

        bint setMaximumSolutions(int value)
        int getMaximumSolutions()
        
        CppOsiSolverInterface* solver()

cdef class CyCbcModel:
    cdef CppICbcModel* CppSelf
    cdef object cyLPModel
    cdef object clpModel
    cdef object cutGenerators
    cdef setCppSelf(self, CppICbcModel* cppmodel)
    cdef setClpModel(self, clpmodel)
    cpdef addCutGenerator(self, CyCglCutGenerator generator,
                        howOften=*, name=*, normal=*, atSolution=*,
                        infeasible=*, howOftenInSub=*, whatDepth=*,
                        whatDepthInSub=*)
