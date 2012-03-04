from cpython.ref cimport PyObject

cdef extern from "ICoinIndexedVector.hpp":
    cdef cppclass CppCoinIndexedVector "ICoinIndexedVector":
        int getNumElements()
        PyObject* getIndicesNPArray()
        PyObject* getDenseVectorNPArray()
        double & operator[](int)
        double getItem(int n)
        void setItem(int n, double value)
        void insert(int ind, double element)
        void reserve(int n)
        void clear()
        void assign(PyObject*, PyObject*)
        void Print()
    CppCoinIndexedVector *new_CppCoinIndexedVector "new ICoinIndexedVector" ()


cdef class CyCoinIndexedVector:
    cpdef CppCoinIndexedVector *CppSelf
    cpdef reserve(self, n)
    cpdef assign(self, ind, other)
    #cpdef getIndices(self)
    #cpdef getElements(self)
    #cpdef getNumElements(self)
