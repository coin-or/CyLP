# cython: embedsignature=True


import pyximport
pyximport.install()


cimport cpython.ref as cpy_ref
#from cpython.ref cimport PyObject
from cpython cimport PyObject, Py_INCREF

from cylp.cy.CyClpPrimalColumnPivotBase cimport CyClpPrimalColumnPivotBase
from cylp.cy.CyClpDualRowPivotBase cimport CyClpDualRowPivotBase
#from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector, CppCoinIndexedVector
from cylp.cy.CyCoinModel cimport CyCoinModel, CppCoinModel
from cylp.cy.CyCoinPackedMatrix cimport CyCoinPackedMatrix, CppCoinPackedMatrix
from cylp.cy.CyCbcModel cimport CyCbcModel, CppICbcModel
from cylp.python.modeling.CyLPModel import CyLPModel
from cylp.cy.CyCoinIndexedVector cimport CyCoinIndexedVector, CppCoinIndexedVector

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "IClpPrimalColumnPivotBase.h" namespace "IClpSimplex":
    cdef enum Status:
        isFree = 0x00,
        basic = 0x01,
        atUpperBound = 0x02,
        atLowerBound = 0x03,
        superBasic = 0x04,
        isFixed = 0x05

cdef extern from "ClpPrimalColumnPivot.hpp":
    cdef cppclass CppClpPrimalColumnPivot "ClpPrimalColumnPivot":
        pass
    CppClpPrimalColumnPivot *new_ClpPrimalColumnPivot \
                                    "new ClpPrimalColumnPivot" ()

cdef extern from "ClpDualRowPivot.hpp":
    cdef cppclass CppClpDualRowPivot "ClpDualRowPivot":
        pass
    CppClpDualRowPivot *new_ClpDualRowPivot \
                                    "new ClpDualRowPivot" ()

cdef extern from "IClpSimplex.hpp":

    ctypedef int (*runIsPivotAcceptable_t)(void* obj)
    ctypedef int (*varSelCriteria_t)(void* obj, int varInd)
    cdef double cdot(CppCoinIndexedVector* pv1, CppCoinIndexedVector* pv2)

    cdef cppclass CppIClpSimplex "IClpSimplex":
        CppIClpSimplex(PyObject* obj,
                       runIsPivotAcceptable_t runIsPivotAcceptable,
                       varSelCriteria_t runVarSelCriteria)

        void setInteger(int index)
        void copyInIntegerInformation(char* information)

        void setCriteria(varSelCriteria_t vsc)
        void setPrimalColumnPivotAlgorithm(CppClpPrimalColumnPivot* choice)
        void setDualRowPivotAlgorithm(CppClpDualRowPivot* choice)
        int readMps(char*, int keepNames, int ignoreErrors)
        void loadQuadraticObjective(CppCoinPackedMatrix* matrix)
        CppCoinPackedMatrix* getMatrix()
        int primal(int ifValuesPass, int startFinishOptions)
        int dual(int ifValuesPass, int startFinishOptions)
        int initialSolve(int presolveType)
        int initialPrimalSolve()
        int initialDualSolve()
        void setPerturbation(int value)
        double* djRegion()
        int getNumCols()
        int getNumRows()
        Status getStatus(int sequence)
        void setStatus(int sequence, Status newstatus)
        double objectiveValue()
        int numberIterations()
        int* QP_ComplementarityList
        int* QP_BanList
        int QP_ExistsBannedVariable

        void useCustomPrimal(int customPrimal)
        int getUseCustomPrimal()

        void setObjectiveCoefficient(int elementIndex, double elementValue )
        void resize(int newNumberRows, int newNumberColumns)

        void setComplementarityList(int* cl)

        void addRow(int numberInRow,
                    int * columns,
                    double * elements,
                    double rowLower,
                    double rowUpper)

        void addColumn(int numberInColumn,
                int * rows,
                double * elements,
                double columnLower,
                double  columnUpper,
                double  objective)

        #number is the number of columns to be added
        void addColumns(int number,
                        double * columnLower,
                        double * columnUpper,
                        double * objective,
                        int * columnStarts,
                        int * rows,
                        double * elements)

        void deleteColumns(int number, int * which)
        void deleteRows(int number, int * which)

        #number is the number of rows to be added
        void addRows(int number,
                        double * rowLower,
                        double * rowUpper,
                        int * rowStarts,
                        int * columns,
                        double * elements)

        void getBasics(int* index)
        void getBInvACol(int col, double* vec)
        void getBInvCol(int col, double* vec)
        void getACol(int ncol, CppCoinIndexedVector * colArray)
        void getRightHandSide(double* righthandside)

        void setColumnUpper(int elementIndex, double elementValue)
        void setColumnLower(int elementIndex, double elementValue)
        void setRowUpper(int elementIndex, double elementValue)
        void setRowLower(int elementIndex, double elementValue)

        double* primalColumnSolution()
        double* dualColumnSolution()
        double* primalRowSolution()
        double* dualRowSolution()
        int status()

        bint flagged(int sequence)
        void setFlagged(int varInd)

        double largestDualError()

        int pivotRow()
        void setPivotRow(int v)

        int sequenceIn()
        void setSequenceIn(int v)

        double currentDualTolerance()
        double dualTolerance()
        void setDualTolerance(double value)
        double primalTolerance()
        void setPrimalTolerance(double value)

        double* rowUpper()
        double* rowLower()
        int numberRows()
        int* ComplementarityList()
        int * pivotVariable()
        #void computeDuals()

        #methods that return nunmpy arrays from c (double*  ,...)
        PyObject* getReducedCosts()
        void setReducedCosts(double* rc)
        PyObject* getStatusArray()
        PyObject* getComplementarityList()
        PyObject* getPivotVariable()

        PyObject* getPrimalRowSolution()
        PyObject* getPrimalColumnSolution()
        PyObject* getPrimalColumnSolutionAll()
        PyObject* getSolutionRegion()
        PyObject* getCostRegion()
        PyObject* getDualRowSolution()
        PyObject* getDualColumnSolution()

        PyObject* filterVars(PyObject*)

        void vectorTimesB_1(CppCoinIndexedVector* vec)
        void transposeTimesSubset(int number, int* which,
                                  double* pi, double* y)
        void transposeTimes(CppIClpSimplex * model, double scalar,
                                 CppCoinIndexedVector * x,
                                 CppCoinIndexedVector * y,
                                 CppCoinIndexedVector * z)
        void transposeTimesSubsetAll(int number, long long int* which,
                                     double* pi, double* y)
        int updateColumnFT(CppCoinIndexedVector* spare, CppCoinIndexedVector* updatedColumn)
        int updateColumnTranspose(CppCoinIndexedVector* regionSparse1,
                                  CppCoinIndexedVector* regionSparse2)

        CppIClpSimplex* preSolve(CppIClpSimplex* si,
                              double feasibilityTolerance,
                              bint keepIntegers,
                              int numberPasses,
                              bint dropNames,
                              bint doRowObjective)
        void postSolve(bint updateStatus)
        int dualWithPresolve(CppIClpSimplex* si,
                              double feasibilityTolerance,
                              bint keepIntegers,
                              int numberPasses,
                              bint dropNames,
                              bint doRowObjective)
        int primalWithPresolve(CppIClpSimplex* si,
                              double feasibilityTolerance,
                              bint keepIntegers,
                              int numberPasses,
                              bint dropNames,
                              bint doRowObjective)

        int loadProblem(CppCoinModel * modelObject, int tryPlusMinusOne)
        void loadProblem(CppCoinPackedMatrix* matrix,
		                  double* collb,  double* colub,
		                  double* obj,
		                  double* rowlb,  double* rowub,
		                  double * rowObjective)

        void setComplement(int var1, int var2)

        void replaceMatrix(CppCoinPackedMatrix* matrix, bint deleteCurrent)

        double getCoinInfinity()

        void setColumnUpperArray(double* columnUpper)
        void setColumnLowerArray(double* columnLower)
        void setColumnUpperSubset(int n, int* indOfind, int *indices, double* values)
        void setColumnLowerSubset(int n, int* indOfind, int *indices, double* values)
        void setColumnUpperFirstElements(int n, double* values)
        void setColumnLowerFirstElements(int n, double* values)
        void setRowUpperArray(double* rowUpper)
        void setRowLowerArray(double* rowLower)
        void setObjectiveArray(double* objective, int numberColumns)
        PyObject* getRowLower()
        PyObject* getRowUpper()
        PyObject* getLower()
        PyObject* getUpper()
        PyObject* getColLower()
        PyObject* getColUpper()
        PyObject* getObjective()
        PyObject* getColumnScale()
        PyObject* getRowScale()
        int writeMps(char* filename, int formatType, int numberAcross,
                     double objSense)
        int readLp(char *filename, double epsilon)

        void setVariableName(int varInd, char* name)
        void setConstraintName(int constInd, char* name)
        vector[string] getVariableNames()

        int partialPrice(int start, int end, int* numberWanted)

        int varIsFree(int ind)
        int varBasic(int ind)
        int varAtUpperBound(int ind)
        int varAtLowerBound(int ind)
        int varSuperBasic(int ind)
        int varIsFixed(int ind)

        #int argWeightedMax(PyObject* arr, PyObject* whr, double weight)
        int argWeightedMax(PyObject* arr, PyObject* arr_ind, PyObject* w,
                            PyObject* w_ind)

        CppICbcModel* getICbcModel()
        void writeLp(char *filename, char *extension,
                       double epsilon, int numberAcross,
                       int decimals, double objSense,
                       bint useRowNames)

        void setMaxNumIteration(int m)
        int maximumIterations()

        #Osi
        void setBasisStatus(int* cstat, int* rstat)
        void getBasisStatus(int* cstat, int* rstat)

        void setLogLevel(int value)
        int logLevel()

        void setObjectiveOffset(double value)
        double objectiveOffset()

        bint automaticScaling()
        void setAutomaticScaling(bint value)

        void scaling(int mode)
        int scalingFlag()

        void setInfeasibilityCost(double value)
        double infeasibilityCost()

        int numberPrimalInfeasibilities()

        bint isInteger(int index)
        PyObject* getIntegerInformation()

        double optimizationDirection()
        void setOptimizationDirection(double value)

cdef class CyClpSimplex:
    '''
    This is the documentation of CyClpSimpelx in the pyx class
    '''

    cdef CppIClpSimplex *CppSelf
    cdef vars
    cdef object varSelCriteria
    cdef CyCoinModel coinModel
    cdef object cyLPModel
    cdef CyCbcModel cbcModel
    cdef object _Hessian

    #cdef void prepareForCython(self, int useCustomPrimal)
    cdef setCppSelf(self,  CppIClpSimplex* s)

    cdef CyClpPrimalColumnPivotBase cyPivot
    cdef CyClpDualRowPivotBase cyDualPivot
    #cdef CppICbcModel* cbcModel
    #cdef object nodeCompareObject
    #cdef cbcModelExists
    #cdef object pivotMethodObject
    #cdef object isPivotAcceptable_func

    cpdef int readMps(self, filename, int keepNames=*,
                int ignoreErrors=*) except *

    cdef setPrimalColumnPivotAlgorithm(self, void* choice)
    cdef setDualRowPivotAlgorithm(self, void* choice)
    cdef double* primalColumnSolution(self)
    cdef double* dualColumnSolution(self)
    cdef double* primalRowSolution(self)
    cdef double* dualRowSolution(self)
    cdef double* rowLower(self)
    cdef double* rowUpper(self)

    #methods that return numpy arrays from c (double*  ,...)
    cpdef getReducedCosts(self)
    cpdef getStatusArray(self)
    cpdef getComplementarityList(self)
    cpdef getPivotVariable(self)

    cpdef filterVars(self, inds)

    cpdef CLP_getVarStatus(self, int sequence)
    cpdef CLP_setVarStatus(self, int sequence, int status)

    cdef primalRow(self, CppCoinIndexedVector*,
                                CppCoinIndexedVector*,
                                CppCoinIndexedVector*,
                                CppCoinIndexedVector*,
                                int)

    #cdef void CLP_getBInvACol(self, int col, double* vec)
    #cdef void CLP_getRightHandSide(self, double* righthandside)

    cpdef getACol(self, int ncol, CyCoinIndexedVector colArray)

    #cdef void CLP_setComplementarityList(self, int*)
    cdef int* ComplementarityList(self)
    cdef int* pivotVariable(self)

    cpdef vectorTimesB_1(self, CyCoinIndexedVector vec)

    #cpdef int loadProblem(self, CyCoinModel modelObject, int tryPlusMinusOne=*)

    #cpdef getPrimalConstraintSolution(self)
    #cpdef getPrimalVariableSolution(self)
    #cpdef getDualConstraintSolution(self)
    #cpdef getDualVariableSolution(self)
    #cpdef createComplementarityList(self)

    cpdef setVariableName(self, varInd, name)
    cpdef setConstraintName(self, constInd, name)

cdef class VarStatus:
    pass
cpdef cydot(CyCoinIndexedVector v1, CyCoinIndexedVector v2)

