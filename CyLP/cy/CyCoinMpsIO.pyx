'''
This module interface COIN-OR's ``CoinMpsIO``. When you call
:func:`CyLP.cy.CyClpSimplex.readMps` then ``CoinMpsIO``'s ``readMps`` is
called.  The main reason why CyLP interfaces this class is to be able to read
an ``mps`` file without creating a Simplex object. This way it is possible to
read a QP using CoinMpsIO and work on the elements of the problem, e.g. the
Hessian,...
'''
# cython: embedsignature=True


from scipy import sparse
from CyLP.py.utils.sparseUtil import csc_matrixPlus, csr_matrixPlus

cdef class CyCoinMpsIO:
    def __cinit__(self):
        self.CppSelf = new CppICoinMpsIO()
        self.Hessian = 0

    def readMps(self, filename):
        '''
        Read an mps file. Check if the file is a QP symmetrisize its Hessian
        and store it.

        >>> import numpy as np
        >>> from CyLP.cy import CyCoinMpsIO
        >>> from CyLP.cy.CyCoinMpsIO import getQpsExample
        >>> problem = CyCoinMpsIO()
        >>> problem.readMps(getQpsExample())
        0
        >>> problem.nVariables
        5
        >>> problem.nConstraints
        5
        >>> signs = problem.constraintSigns
        >>> [chr(i) for i in signs] == problem.nConstraints * ['G']
        True
        >>> c = problem.matrixByRow
        >>> (abs(c.elements - 
        ...     np.array([-1., -1., -1., -1., -1.,  10.,  10.,  -3.,
        ...                5., 4.,  -8., 1., -2., -5., 3., 8., -1., 2.,
        ...                5., -3.,  -4.,  -2., 3., -5., 1.])) < 
        ...                            10 ** -8).all()
        True
        >>> (c.indices ==
        ...       np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
        ...                 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)).all()
        True
        >>> (c.vectorStarts ==
        ...        np.array([0, 5, 10, 15, 20, 25], dtype=np.int32)).all()
        True
        >>> (problem.rightHandSide ==
        ...        np.array([-5., 20., -40., 11., -30.])).all()
        True
        >>> H = problem.Hessian.todense()
        >>> (abs(H -
        ... np.matrix([[20394., -24908., -2026., 3896., 658.],
        ...            [-24908., 41818., -3466., -9828., -372.],
        ...            [-2026., -3466., 3510., 2178., -348.],
        ...            [3896., -9828., 2178., 3030., -44.],
        ...            [658., -372., -348., -44., 54.]])) < 
        ...                            10 ** -8).all()
        True

        '''

        ret = self.CppSelf.readMps(filename)
        if ret == 0 and self.CppSelf.IreadQuadraticMps(NULL, 0) == 0:
            start = self.QPColumnStarts
            col = self.QPColumns
            el = self.QPElements

            n = self.nVariables

            #Hessian = csc_matrixPlus((el,col,start), shape=(n, n))

            ### To make a csc Hessian symmetric
            #n = self.getNumCols()

            #for j in range(n):
            #   for i in range(Hessian.indptr[j],Hessian.indptr[j+1]):
            #       iRow = Hessian.indices[i]
            #       Hessian[j, iRow] = Hessian[iRow, j]
            #print Hessian.todense()

            Hessian = csr_matrixPlus((el, col, start), shape=(n, n))
            
            m = self.nConstraints
            
#            for i in xrange(n):
#                for j in xrange(Hessian.indptr[i], Hessian.indptr[i + 1]):
#                    jCol = Hessian.indices[j]
#                    Hessian[jCol, i] = Hessian[i, jCol]
            Hessian = Hessian + Hessian.T
            for i in xrange(n):
                Hessian[i, i] /= 2
            self.Hessian = Hessian

        return ret

    def readQuadraticMps(self, filename, checkSymmetry):
        return self.CppSelf.IreadQuadraticMps(NULL, checkSymmetry)

    property Hessian:
        def __get__(self):
            return self.Hessian

        def __set__(self, h):
            self.Hessian = h

    property variableLower:
        def __get__(self):
            return <object>self.CppSelf.np_getColLower()

    property variableUpper:
        def __get__(self):
            return <object>self.CppSelf.np_getColUpper()

    property constraintSigns:
        def __get__(self):
            return <object>self.CppSelf.np_getRowSense()

    property rightHandSide:
        def __get__(self):
            return <object>self.CppSelf.np_getRightHandSide()

    property constraintRange:
        def __get__(self):
            return <object>self.CppSelf.np_getRowRange()

    property constraintLower:
        def __get__(self):
            return <object>self.CppSelf.np_getRowLower()

    property constraintUpper:
        def __get__(self):
            return <object>self.CppSelf.np_getRowUpper()

    property objCoefficients:
        def __get__(self):
            return <object>self.CppSelf.np_getObjCoefficients()

    property integerColumns:
        def __get__(self):
            return <object>self.CppSelf.np_integerColumns()

    property QPColumnStarts:
        def __get__(self):
            return <object>self.CppSelf.getQPColumnStarts()

    property QPColumns:
        def __get__(self):
            return <object>self.CppSelf.getQPColumns()

    property QPElements:
        def __get__(self):
            return <object>self.CppSelf.getQPElements()

    property matrixByRow:
        def __get__(self):
            cdef CppCoinPackedMatrix* m = self.CppSelf.IgetMatrixByRow()
            cym = CyCoinPackedMatrix()
            cym.CppSelf = m
            return cym

    property matrixByCol:
        def __get__(self):
            cdef CppCoinPackedMatrix* m = self.CppSelf.IgetMatrixByCol()
            cym = CyCoinPackedMatrix()
            cym.CppSelf = m
            return cym

    property objectiveOffset:
        def __get__(self):
            return self.CppSelf.getObjectiveOffset()

    property nVariables:
        def __get__(self):
            return self.CppSelf.getNumCols()

    property nConstraints:
        def __get__(self):
            return self.CppSelf.getNumRows()


def getQpsExample():
    '''
    Return full path to a QPS example file for doctests
    '''
    import os
    import inspect
    curpath = os.path.dirname(inspect.getfile(inspect.currentframe()))
    return os.path.join(curpath, '../input/hs268.qps')


#   cdef int CLP_readQuadraticMps(self, char* filename,
#               int * columnStart, int * column2, double * elements,
#               int checkSymmetry):
#       return self.CppSelf.readQuadraticMps(filename, columnStart,
#                                        column2, elements, checkSymmetry)

#   def readQuadraticMps(self,
#                           filename,
#                           np.ndarray[np.int32_t,ndim=1] columnStart,
#                           np.ndarray[np.int32_t,ndim=1] column2,
#                           np.ndarray[np.double_t,ndim=1] elements,
#                           checkSymmetry):
#       return self.CLP_readQuadraticMps(filename,
#                                           <int*>columnStart.data,
#                                           <int*>column2.data,
#                                           <double*>elements.data,
#                                           checkSymmetry)
