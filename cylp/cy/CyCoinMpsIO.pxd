from cpython.ref cimport PyObject
cimport numpy as np
from cylp.cy.CyCoinPackedMatrix cimport CyCoinPackedMatrix, CppCoinPackedMatrix

cdef extern from "ICoinMpsIO.hpp":
    cdef cppclass CppICoinMpsIO "ICoinMpsIO":
        #PyObject* getIndicesNPArray()
        int readMps(char *filename)
        int getNumCols()
        int getNumRows()
        int getNumElements()
        PyObject* np_getColLower()
        PyObject* np_getColUpper()
        PyObject* np_getRowSense()
        PyObject* np_getRightHandSide()
        PyObject* np_getRowRange()
        PyObject* np_getRowLower()
        PyObject* np_getRowUpper()
        PyObject* np_getObjCoefficients()
        bint isContinuous(int colNumber)
        bint isInteger(int columnNumber)
        PyObject* np_integerColumns()
        char* rowName(int index)
        char* columnName(int index)
        int rowIndex(char * name)
        int columnIndex(char * name)
        char* getProblemName()

        CppCoinPackedMatrix * IgetMatrixByRow()
        CppCoinPackedMatrix * IgetMatrixByCol()

        PyObject* getQPColumnStarts()
        PyObject* getQPColumns()
        PyObject* getQPElements()

        double getObjectiveOffset()

        #int readQuadraticMps(char* filename,
        #       int * columnStart, int * column2, double * elements,
        #       int checkSymmetry)
        int IreadQuadraticMps(char* filename, int checkSymmetry)

    CppICoinMpsIO *new_CppICoinMpsIO "new CppICoinMpsIO" ()


cdef class CyCoinMpsIO:
    cdef CppICoinMpsIO *CppSelf
    cdef Hessian
#    cpdef  getColLower(self)
#    cpdef  getColUpper(self)
#    cpdef getRowSense(self)
#    cpdef getRightHandSide(self)
#    cpdef getRowRange(self)
#    cpdef getRowLower(self)
#    cpdef getRowUpper(self)
#    cpdef getObjCoefficients(self)
#    cpdef integerColumns(self)
#
#    cpdef getQPColumnStarts(self)
#    cpdef getQPColumns(self)
#    cpdef getQPElements(self)
#    cpdef getHessian(self)
#
#    cpdef getMatrixByRow(self)
#    cpdef getMatrixByCol(self)


    #cdef int CLP_readQuadraticMps(self, char* filename,
    #           int * columnStart, int * column2, double * elements,
    #           int checkSymmetry)

