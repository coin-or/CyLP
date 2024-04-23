# cython: embedsignature=True

cdef class CyCoinIndexedVector:
    '''
    ``CyCoinIndexedVector`` interfaces ``CoinIndexedVector``.

    **Usage**

    >>> from cylp.cy import CyCoinIndexedVector
    >>> vector = CyCoinIndexedVector()
    >>> vector.reserve(5)
    >>> vector[3] = 4
    >>> vector[3]
    4.0
    >>> vector[[1, 3, 4]] = 2.1
    >>> [vector[i] for i in [1, 3, 4]] == 3 * [2.1]
    True
    >>> vector[[0, 2]] = [-1, -2]
    >>> [vector[i] for i in [0, 2]] == [-1, -2]
    True
    '''

    def __cinit__(self):
        self.CppSelf = new_CppCoinIndexedVector()

    #def __dealloc__(self):
    #    del self.CppSelf

    cdef setCppSelf(self, CppCoinIndexedVector* s):
        del self.CppSelf
        self.CppSelf = s

    def Print(self):
        self.CppSelf.Print()

    cpdef reserve(self, n):
        self.CppSelf.reserve(n)

    def __getitem__(self, n):
        return self.CppSelf.getItem(n)

    def __setitem__(self, key, value):
        self.assign(key, value)
        #self.CppSelf.insert(key, value)
        #self.CppSelf.setItem(key, value)

    #def __setslice__(self, rg, val):
        #print(rg, val)

    def clear(self):
        self.CppSelf.clear()

    def empty(self):
        self.CppSelf.empty()

    cpdef assign(self, ind, other):
        self.CppSelf.assign(<PyObject*>ind, <PyObject*>other)

    property indices:
        def __get__(self):
            return <object>self.CppSelf.getIndicesNPArray()

    property elements:
        def __get__(self):
            return <object>self.CppSelf.getDenseVectorNPArray()

    property nElements:
        def __get__(self):
            return self.CppSelf.getNumElements()

        def __set__(self, value):
            self.CppSelf.setNumElements(value)

    property isInPackedMode:
        def __get__(self):
            return self.CppSelf.packedMode()
