# cython: embedsignature=True

import numpy as np
cimport numpy as np


cdef class CyCoinPackedMatrix:
    '''
    ``CyCoinPackedMatrix`` interfaces ``CoinPackedMatrix``

    **Usage Example**

    >>> import numpy as np
    >>> from CyLP.cy import CyCoinPackedMatrix
    >>> rows = np.array([1, 3, 4], np.int32)
    >>> cols = np.array([0, 2, 1], np.int32)
    >>> elements = np.array([1.5, -1, 2])
    >>> # colOrdered is True if we store the matrix by column (csc)
    >>> m = CyCoinPackedMatrix(colOrdered=True, rowIndices=rows,
    ...                           colIndices=cols, elements=elements)
    >>> m.majorDim
    3
    >>> m.minorDim
    5
    '''
    def __cinit__(self, colOrdered=True,
                        np.ndarray[np.int32_t, ndim=1] rowIndices=None,
                        np.ndarray[np.int32_t, ndim=1] colIndices=None,
                        np.ndarray[np.double_t, ndim=1] elements=None):
        if rowIndices == None:
            self.CppSelf = new_CppCoinPackedMatrix()
        else:
            self.CppSelf = new CppCoinPackedMatrix(colOrdered,
                                    <int*>rowIndices.data,
                                    <int*> colIndices.data,
                                    <double*> elements.data,
                                    len(rowIndices))

    property indices:
        def __get__(self):
            return <object>self.CppSelf.np_getIndices()

    property elements:
        def __get__(self):
            return <object>self.CppSelf.np_getElements()

    property vectorStarts:
        def __get__(self):
            return <object>self.CppSelf.np_getVectorStarts()

    #cpdef reserve(self, n):
    #   self.CppSelf.reserve(n)

    property nElements:
        def __get__(self):
            return self.CppSelf.getNumElements()

    property majorDim:
        def __get__(self):
            return self.CppSelf.getMajorDim()

    property minorDim:
        def __get__(self):
            return self.CppSelf.getMinorDim()
    
    def reserve(self,  newMaxMajorDim,  newMaxSize,  create=0):
        self.CppSelf.reserve(newMaxMajorDim, newMaxSize, create)

    def appendRow(self, np.ndarray[np.int32_t, ndim=1] vecInd=None,
                  np.ndarray[np.double_t, ndim=1] elements=None):
       
        cdef int* i
        cdef double* d
        if vecInd:
            self.CppSelf.appendRow(len(elements), <int*>vecInd.data, 
                                            <double*>elements.data)
        else:
            self.CppSelf.appendRow(0, i, d) 

    def appendCol(self, np.ndarray[np.int32_t, ndim=1] vecInd=None,
                  np.ndarray[np.double_t, ndim=1] elements=None):
       
        cdef int* i
        cdef double* d
        if vecInd:
            self.CppSelf.appendCol(len(elements), <int*>vecInd.data, 
                                            <double*>elements.data)
        else:
            self.CppSelf.appendCol(0, i, d) 
    #def __getitem__(self, n):
    #   return self.CppSelf.getItem(n)

    #def __setitem__(self, key, value):
    #   self.CppSelf.setItem(key, value)
