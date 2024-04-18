# cython: embedsignature=True

cdef class CyCoinModel:
    '''

    >>> import numpy as np
    >>> from cylp.cy import CyCoinModel, CyClpSimplex
    >>> m = CyCoinModel()
    >>>
    >>> m.addVariable(3, np.array(
    ...                 [0, 1, 2], np.int32),
    ...                 np.array([1., 1., 1.], np.double), 0, 10, 5)
    >>>
    >>> m.addVariable(2, np.array(
    ...                 [1,2], np.int32),
    ...                 np.array([5, 2.], np.double), 0, 10, 2)
    >>>
    >>> # Add bounds for the three constraints (we have two variables)
    >>> m.setConstraintLower(0, 2.3)
    >>> m.setConstraintLower(1, 4.5)
    >>> m.setConstraintLower(0, 1.5)
    >>>
    >>> # Add a 4th constraint
    >>> m.addConstraint(2,
    ...                     np.array([0, 1], np.int32),
    ...                     np.array([1., 1.], np.double), 2, 7)
    >>>
    >>> s = CyClpSimplex()
    >>> # Load the problem from the CyCoinModel
    >>> s.loadProblemFromCyCoinModel(m)
    0
    >>> s.primal()
    'optimal'
    >>> abs(s.objectiveValue - 8.7) < 10 ** -7
    True

    '''

    def __cinit__(self):
        self.CppSelf = new CppCoinModel()

    def __dealloc__(self):
        del self.CppSelf

#   cdef void CLP_addColumn(self, int numberInColumn,
#                           int * rows,
#                           double * elements,
#                           double columnLower,
#                           double columnUpper,
#                           double objectiveValue,
#                           char * name,
#                           int isInteger):
#       self.CppSelf.addColumn(numberInColumn, rows, elements,
#                              columnLower,columnUpper,
#                              objectiveValue,name, isInteger)

    cdef void CLP_addColumn(self, int numberInColumn,
                            int * rows,
                            double * elements,
                            double columnLower,
                            double columnUpper,
                            double objectiveValue):
        self.CppSelf.addColumn(numberInColumn, rows, elements,
                               columnLower, columnUpper, objectiveValue)

#   def addVariable(self, numberInColumn,
#                       np.ndarray[np.int32_t, ndim=1] rows,
#                       np.ndarray[np.double_t, ndim=1] elements,
#                       columnLower,
#                       columnUpper,
#                       objective,
#                       name,
#                       isInteger):
#        TODO: This makes adding a column real slower,
#         but it is better than a COIN EXCEPTION!
#        for r in rows:
#           if r >= self.getNumRows():
#               raise Exception('CyClpSimplex.pyx:addColumn: Row number ' \
#                                '%d should be less than row size %d' %
#                                (r, self.getNumRows()))
#       self.CLP_addColumn(numberInColumn, <int*>rows.data,
#                            <double*> elements.data, columnLower, columnUpper,
#                            objective, name, isInteger)

    def addVariable(self, numberInColumn,
                        np.ndarray[np.int32_t, ndim=1] rows,
                        np.ndarray[np.double_t, ndim=1] elements,
                        columnLower,
                        columnUpper,
                        objective):
        self.CLP_addColumn(numberInColumn, <int*>rows.data,
                          <double*> elements.data, columnLower,
                          columnUpper, objective)

#   cdef void CLP_addRow(self, int numberInRow, int * columns,
#                       double * elements, double rowLower,
#                       double rowUpper, char * name):
#       self.CppSelf.addRow(numberInRow, columns, elements, rowLower,
#                            rowUpper, name)

    cdef void CLP_addRow(self, int numberInRow, int * columns,
                        double * elements, double rowLower,
                        double rowUpper):
        self.CppSelf.addRow(numberInRow, columns, elements, rowLower, rowUpper)

#   def addConstraint(self, numberInRow,
#                   np.ndarray[np.int32_t, ndim=1] columns,
#                   np.ndarray[np.double_t, ndim=1] elements,
#                   rowLower,
#                   rowUpper,
#                   name = ''):
#       self.CLP_addRow(numberInRow, <int*>columns.data,
#                        <double*>elements.data, rowLower, rowUpper, name)

    def addConstraint(self, numberInRow,
                    np.ndarray[np.int32_t, ndim=1] columns,
                    np.ndarray[np.double_t, ndim=1] elements,
                    rowLower,
                    rowUpper):
        self.CLP_addRow(numberInRow, <int*>columns.data,
                        <double*>elements.data, rowLower, rowUpper)

    def setVariableLower(self, ind, val):
        self.CppSelf.setColumnLower(ind, val)

    def setVariableUpper(self, ind, val):
        self.CppSelf.setColumnUpper(ind, val)

    def setConstraintLower(self, ind, val):
        self.CppSelf.setRowLower(ind, val)

    def setConstraintUpper(self, ind, val):
        self.CppSelf.setRowUpper(ind, val)

    def setObjective(self, varInd, val):
        self.CppSelf.setObjective(varInd, val)

    def getNumVariables(self):
        return self.CppSelf.numberColumns()

    def getNumConstraints(self):
        return self.CppSelf.numberRows()
