from cpython.ref cimport PyObject

cdef extern from "ICoinPackedMatrix.hpp":
    cdef cppclass CppCoinPackedMatrix "ICoinPackedMatrix":
        int getNumElements()
        PyObject* np_getIndices()
        PyObject* np_getElements()
        PyObject* np_getVectorStarts()
        int getMajorDim()
        int getMinorDim()
        void reserve(int newMaxMajorDim, int newMaxSize, bint create)
        CppCoinPackedMatrix(bint colordered, int* rowIndices,
                                    int* colIndices, double* elements,
                                    int numels)
        void appendRow(int vecsize, int *vecind, double *vecelem)
        void appendCol(int vecsize, int *vecind, double *vecelem)
        bint isColOrdered()
        void dumpMatrix(char* fname)
        PyObject * np_getMajorIndices()
        bint hasGaps()
        void removeGaps(double removeValue)
        #void setItem(int n, double value)
        #void reserve(int n)
    CppCoinPackedMatrix *new_CppCoinPackedMatrix "new ICoinPackedMatrix" ()


cdef class CyCoinPackedMatrix:
    cdef CppCoinPackedMatrix* CppSelf
    #cpdef getIndices(self)
    #cpdef getElements(self)
    #cpdef getNumElements(self)
    #cpdef getVectorStarts(self)
    #cpdef getMajorDim(self)
    #cpdef getMinorDim(self)
