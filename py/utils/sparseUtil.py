'''
sparseUtil provides useful sparse matrix operations for incremental
construction of large sparse matrices from its block elements. Class
:class:`csc_matrixPlus` derives from ``sparse.csc_matrix`` and makes it
possible to set elements outside the matrix dimensions by resizing
the matrix whenever necessary. Class :class:`csr_matrixPlus` does
the same thing with ``sparse.csr_matrix``.

Function :py:func:`sparseConcat` concatenates two sparse matrices
regardless of their dimension alignments. Fills with zeros where necessary.

'''

from scipy import sparse
import numpy as np



class csc_matrixPlus(sparse.csc_matrix):
    def __init__(self, arg1, shape=None, dtype=None,
                 copy=False, fromMatrix=None):
        sparse.csc_matrix.__init__(self, arg1, shape, dtype, copy)
        if fromMatrix:
            self.__dict__.update(fromMatrix.__dict__)
        from CyLP.py.modeling import CyLPExpr as CyLPExpr 
        self.CyLPExpr = CyLPExpr

    def __setitem__(self, (iRow, iCol), val):
        '''
        Set the item in row ``i`` and column ``j`` to ``val``.
        Increases matrix's size if necessary

        **Usage**

        >>> from CyLP.py.utils.sparseUtil import csc_matrixPlus
        >>> import numpy as np
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> s = csc_matrixPlus((data, indices, indptr), shape=(3, 3))
        >>> s[2, 5] = 11
        >>> s.todense()
        matrix([[ 1,  0,  4,  0,  0,  0],
                [ 0,  0,  5,  0,  0,  0],
                [ 2,  3,  6,  0,  0, 11]])

        '''
        if not isinstance(val, (int, long, float)):
            return sparse.csc_matrix.__setitem__(self, (iRow, iCol), val)
            
        nCols = self.shape[1]

        #update shape if nec.
        if iRow >= self.shape[0]:
            self._shape = (iRow + 1, self.shape[1])

        if iCol < nCols:
            for i in range(self.indptr[iCol], self.indptr[iCol + 1]):
                if self.indices[i] == iRow:
                    self.data[i] = val
                    return
            #If we reach here it means that index does NOT exist
            for i in xrange(iCol + 1, nCols + 1):
                self.indptr[i] += 1
            indexOfElement = self.indptr[iCol + 1] - 1
            #If indices is empty
            if indexOfElement == 0:
                self.indices = np.array([iRow], dtype=np.int32)
                self.data = np.array([val])
            else:
                self.indices = np.concatenate((self.indices[:indexOfElement],
                                        np.array([iRow], dtype=np.int32),
                                        self.indices[indexOfElement:]), axis=0)

                self.data = np.concatenate((self.data[:indexOfElement],
                                        np.array([val]),
                                        self.data[indexOfElement:]), axis=0)

        else:
            #We don't have enough columns, increase dimension 1
            self.addColumns(iCol - nCols + 1)
            self.indptr[iCol + 1] += 1
            self.indices = np.concatenate((self.indices, np.array([iRow],
                dtype=np.int32)), axis=0)
            self.data = np.concatenate((self.data, np.array([val],
                dtype=np.int32)), axis=0)
            self._shape = (self._shape[0], iCol + 1)

    def addColumns(self, nCol):
        '''
        Add ``nCol`` columns to the matrix

        **Usage**

        >>> from CyLP.py.utils.sparseUtil import csc_matrixPlus
        >>> s = csc_matrixPlus.getMatrixForTest()
        >>> s.shape
        (3, 3)
        >>> s.addColumns(3)
        >>> s.shape
        (3, 6)

        '''

        nElement = len(self.data)
        a = np.array(nCol * [nElement], dtype=np.int32)
        self.indptr = np.concatenate((self.indptr, a), axis=0)
        self._shape = (self._shape[0], self.shape[1] + nCol)
    
    def __getitem__(self, key):
        ret = sparse.csc_matrix.__getitem__(self, key)
        if isinstance(ret, (int, long, float)):
            return ret
        return csc_matrixPlus(ret)
    
    @property 
    def T(self):
        return csr_matrixPlus(sparse.csc_matrix.transpose(self))
    
    def __le__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__le__(self, other)

    def __ge__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__ge__(self, other)

    def __mul__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__rmul__(self, other)

    def __add__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__add__(self, other)

    def __radd__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__radd__(self, other)

    def __rsub__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__rsub__(self, other)

    def __sub__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csc_matrix.__sub__(self, other)

    @staticmethod
    def getMatrixForTest():
        from CyLP.py.utils.sparseUtil import csr_matrixPlus
        import numpy as np
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        s = csc_matrixPlus((data, indices, indptr), shape=(3, 3))
        return s


class csr_matrixPlus(sparse.csr_matrix):
    def __init__(self, arg1, shape=None, dtype=None,
                 copy=False, fromMatrix=None):
        sparse.csr_matrix.__init__(self, arg1, shape, dtype, copy)
        if fromMatrix:
            self.__dict__.update(fromMatrix.__dict__)
        from CyLP.py.modeling import CyLPExpr as CyLPExpr 
        self.CyLPExpr = CyLPExpr

    def __setitem__(self, (iRow, iCol), val):
        '''
        Sets the item in row ``i`` and col ``j`` to ``val``.
        Increases matrix's ``shape[1]`` if necessary

        **Usage**

        >>> from CyLP.py.utils.sparseUtil import csr_matrixPlus
        >>> import numpy as np
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> s = csr_matrixPlus((data, indices, indptr), shape=(3, 3))
        >>> s[5,2] = 11
        >>> print s.todense()
        [[ 1  0  2]
         [ 0  0  3]
         [ 4  5  6]
         [ 0  0  0]
         [ 0  0  0]
         [ 0  0 11]]

        '''
        if not isinstance(val, (int, long, float)):
            return sparse.csr_matrix.__setitem__(self, (iRow, iCol), val)
        
        nRows = self.shape[0]

        if iCol >= self.shape[1]:
            self._shape = (self.shape[0], iCol + 1)

        if iRow < nRows:
            for i in range(self.indptr[iRow], self.indptr[iRow + 1]):
                if self.indices[i] == iCol:
                    self.data[i] = val
                    return
            #if we reach here it means that index does NOT exist
            for i in xrange(iRow + 1, nRows + 1):
                self.indptr[i] += 1
            indexOfElement = self.indptr[iRow + 1] - 1
            # If indices is empty
            if indexOfElement == 0:
                self.indices = np.array([iCol], dtype=np.int32)
                self.data = np.array([val])
            else:
                self.indices = np.concatenate((self.indices[:indexOfElement],
                                        np.array([iCol], dtype=np.int32),
                                        self.indices[indexOfElement:]), axis=0)
                self.data = np.concatenate((self.data[:indexOfElement],
                                        np.array([val]),
                                        self.data[indexOfElement:]), axis=0)
        else:
            #We don't have enough columns, increase dimension 1
            self.addRows(iRow - nRows + 1)
            self.indptr[iRow + 1] += 1
            self.indices = np.concatenate((self.indices, np.array([iCol],
                dtype=np.int32)), axis=0)
            self.data = np.concatenate((self.data, np.array([val],
                dtype=np.int32)), axis=0)
            self._shape = (iRow + 1, self._shape[1])

    def addRows(self, nRow):
        '''
        Add ``nRow`` rows to the matrix

        **Usage**

        >>> from CyLP.py.utils.sparseUtil import csr_matrixPlus
        >>> s = csr_matrixPlus.getMatrixForTest()
        >>> s.shape
        (3, 3)
        >>> s.addRows(2)
        >>> s.shape
        (5, 3)

        '''

        nElement = len(self.data)
        a = np.array(nRow * [nElement], dtype=np.int32)
        self.indptr = np.concatenate((self.indptr, a), axis=0)
        self._shape = (self._shape[0] + nRow, self.shape[1])

    def __getitem__(self, key):
        ret = sparse.csr_matrix.__getitem__(self, key)
        if isinstance(ret, (int, long, float)):
            return ret
        return csr_matrixPlus(ret)

    @property 
    def T(self):
        return csc_matrixPlus(sparse.csr_matrix.transpose(self))
    
    def __le__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__le__(self, other)

    def __ge__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__ge__(self, other)

    def __mul__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__rmul__(self, other)

    def __add__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__add__(self, other)

    def __radd__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__radd__(self, other)

    def __rsub__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__rsub__(self, other)

    def __sub__(self, other):
        if isinstance(other, self.CyLPExpr):
            return NotImplemented
        return sparse.csr_matrix.__sub__(self, other)

    @staticmethod
    def getMatrixForTest():
        from CyLP.py.utils.sparseUtil import csr_matrixPlus
        import numpy as np
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        return csr_matrixPlus((data, indices, indptr), shape=(3, 3))


def sparseConcat(a, b, how, offset=0):
    '''
    Concatenate two sparse matrices, ``a`` and ``b``, horizontally if
    ``how = 'h'``, and vertically if ``how = 'v'``.
    Add zero rows and columns if dimensions don't align.
    ``offset`` specifies how to align ``b`` along side ``a``. 
    ``offset=-1`` means the greatest possible offset without 
    changeing the dimension.

    **Usage**

    >>> from scipy import sparse
    >>> from CyLP.py.utils.sparseUtil import sparseConcat, csc_matrixPlus
    >>> s1 = csc_matrixPlus.getMatrixForTest()
    >>> s2 = sparse.lil_matrix([[1,0,2],[0,5,0]])
    >>> sparseConcat(s1, s2, 'v').todense()
    matrix([[1, 0, 4],
            [0, 0, 5],
            [2, 3, 6],
            [1, 0, 2],
            [0, 5, 0]])
    >>> sparseConcat(s1, s2, 'h').todense()
    matrix([[1, 0, 4, 1, 0, 2],
            [0, 0, 5, 0, 5, 0],
            [2, 3, 6, 0, 0, 0]])

    If ``a = None`` then return ``b``. This makes possible an incremental
    construction of large sparse matrices from scratch without the hassle
    of the initial value check.

    >>> s3 = None
    >>> ((sparseConcat(s3, s1, 'h').todense() == s1.todense()).all() and
    ...  (sparseConcat(s3, s1, 'v').todense() == s1.todense()).all())
    True

    '''
    if a == None:
        return sparse.coo_matrix(b.copy())
    assert(offset >= -1)

    a = sparse.coo_matrix(a)
    b = sparse.coo_matrix(b)

    if how == 'h':

        if offset == -1:
            assert(a.shape[0] > b.shape[0])
            offset = a.shape[0] - b.shape[0]
        
        
        row = np.concatenate((a.row, b.row + offset), axis=0)
        col = np.concatenate((a.col, b.col + a.shape[1]), axis=0)
        data = np.concatenate((a.data, b.data), axis=0)

        nRows = max(a.shape[0], b.shape[0] + offset)
        a = sparse.coo_matrix((data, (row, col)),
                              shape=(nRows, a.shape[1] + b.shape[1]))

    elif how == 'v':
        row = np.concatenate((a.row, b.row + a.shape[0]), axis=0)
        if offset == -1:
            assert(a.shape[1] > b.shape[1])
            offset = a.shape[1] - b.shape[1]
        col = np.concatenate((a.col, b.col + offset), axis=0)
        data = np.concatenate((a.data, b.data), axis=0)

        nCols = max(a.shape[1], b.shape[1] + offset)
        a = sparse.coo_matrix((data, (row, col)),
                              shape=(a.shape[0] + b.shape[0], nCols))
    return a


def I(n):
    '''
    Return a sparse identity matrix of size *n*
    '''
    if n <= 0:
        return None
    return csc_matrixPlus(sparse.eye(n, n))


