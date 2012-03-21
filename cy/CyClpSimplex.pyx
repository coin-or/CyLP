# cython: profile=True
# cython: embedsignature=True


from exceptions import TypeError
import inspect
import os.path
import numpy as np
cimport numpy as np
cimport cpython.ref as cpy_ref
from CyWolfePivot cimport CyWolfePivot
from CyPEPivot cimport CyPEPivot
from CyPivotPythonBase cimport CyPivotPythonBase
from CyLP.cy cimport CyClpSimplex
from CyLP.cy cimport CyCoinModel
from CyLP.py.utils.sparseUtil import sparseConcat
from CyLP.py.modeling.CyLPModel import CyLPVar
from CyLP.py.pivots.PivotPythonBase import PivotPythonBase
from CyLP.py.modeling.CyLPModel import CyLPModel
from CyLP.cy cimport CyCoinMpsIO

problemStatus = ['optimal', 'primal infeasible', 'dual infeasible',
                'stopped on iterations or time',
                'stopped due to errors',
                'stopped by event handler (virtual int ' \
                                    'ClpEventHandler::event())']


cdef class CyClpSimplex:
    '''
    CyClpSimplex is a Cython interface to CLP.
    Not all methods are available but they are being added gradually.

    Its constructor can create an empty object if no argument is provided.
    However, if a :class:`CyLPModel <CyLP.py.modeling.CyLPModel>` object is
    given then the resulting ``CyClpSimplex`` object will be build from it.
    For an example of the latter case see
    :mod:`CyLP's modeling tool. <CyLP.py.modeling.CyLPModel>`

    .. _simple-run:

    **An easy example of how to read and solve an LP**

    >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getMpsExample
    >>> s = CyClpSimplex()
    >>> f = getMpsExample()
    >>> s.readMps(f)
    0
    >>> s.initialSolve()
    'optimal'

    '''

    def __cinit__(self, cyLPModel=None):
        self.CppSelf = new CppIClpSimplex(<cpy_ref.PyObject*>self,
                                <runIsPivotAcceptable_t>RunIsPivotAcceptable,
                                <varSelCriteria_t>RunVarSelCriteria)
        self.vars = []
        #self.cbcModelExists = False
        self.coinModel = CyCoinModel()

        self.cyLPModel = cyLPModel
        if cyLPModel:
            if isinstance(cyLPModel, CyLPModel):
                self.loadFromCyLPModel(cyLPModel)
            else:
                raise TypeError('Expected a CyLPModel as an argument to ' \
                                'CyLPSimplex constructor. Got %s' %
                                cyLPModel.__class__)

    cdef setCppSelf(self,  CppIClpSimplex* s):
        self.CppSelf = s

    #############################################
    # Properties
    #############################################

    property objective:
        '''
        Set the objective function using this property.
        See the :ref:`modeling example <modeling-usage>`.
        '''
        def __set__(self, obj):
            if self.cyLPModel:
                self.cyLPModel.objective = obj
                self.setObjectiveArray(
                        self.cyLPModel.objective.astype(np.double))
            else:
                raise Exception('To set the objective function of ' \
                                'CyClpSimplex set CyLPSimplex.cyLPModel ' \
                                'first.')

    property iteration:
        '''
        Number of iterations.
        '''
        def __get__(self):
            return self.CppSelf.numberIterations()

    property nRows:
        '''
        Number of rows, constraints.
        '''
        def __get__(self):
            return self.CppSelf.getNumRows()

    property nConstraints:
        '''
        Number of constraints, rows.
        '''
        def __get__(self):
            return self.CppSelf.getNumRows()

    property nVariables:
        '''
        Number of variables, columns.
        '''
        def __get__(self):
            return self.CppSelf.getNumCols()

    property nCols:
        '''
        Number of columns, variables.
        '''
        def __get__(self):
            return self.CppSelf.getNumCols()

#    def getNumRows(self):
#        '''
#        Return number of constraints
#        '''
#        return self.CppSelf.getNumRows()

#    def getNumCols(self):
#        return self.CppSelf.getNumCols()

    property objectiveValue:
        '''
        The objective value. Readonly.
        '''
        def __get__(self):
            return self.CppSelf.objectiveValue()

    property primalVariableSolution:
        '''
        Solution to the primal variables.

        :rtype: Numpy array
        '''
        def __get__(self):
            #if self.cbcModelExists:
            #    return <object>self.cbcModel.getPrimalVariableSolution()
            return <object>self.CppSelf.getPrimalColumnSolution()

    property primalVariableSolutionAll:
        '''
        Solution to the primal variables. Including the slacks.

        :rtype: Numpy array
        '''
        def __get__(self):
            #if self.cbcModelExists:
            #    return <object>self.cbcModel.getPrimalVariableSolution()
            return <object>self.CppSelf.getPrimalColumnSolutionAll()

    property solution:
        '''
        Return the current point.

        :rtype: Numpy array
        '''
        def __get__(self):
            #if self.cbcModelExists:
            #    return <object>self.cbcModel.getPrimalVariableSolution()
            return <object>self.CppSelf.getSolutionRegion()

    property dualVariableSolution:
        '''
        Solution to the dual variables

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getDualColumnSolution()

    property primalConstraintSolution:
        '''
        Solution to the primal slack variables

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getPrimalRowSolution()

    property dualConstraintSolution:
        '''
        Solution to the dual slack variables

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getDualRowSolution()

    property reducedCosts:
        '''
        The reduced costs. A Numpy array.

        :rtype: Numpy array
        '''
        def __get__(self):
            return self.getReducedCosts()

        def __set__(self, np.ndarray[np.double_t, ndim=1] rc):
            self.CppSelf.setReducedCosts(<double*> rc.data)

    cpdef getReducedCosts(self):
        return <object>self.CppSelf.getReducedCosts()

    property status:
        '''
        A Numpy array of all the variables' status
        '''
        def __get__(self):
            return self.getStatusArray()

    cpdef getStatusArray(self):
        return <object>self.CppSelf.getStatusArray()

    property freeOrSuperBasicVarInds:
        '''
        The index set of variables that are *free* or *superbasic*.
        '''
        def __get__(self):
            status = self.status
            return np.where((status & 7 == 4) | (status & 7 == 0))[0]

    property notBasicOrFixedOrFlaggedVarInds:
        '''
        The index set of variables that are not *basic* or *fixed*.
        '''
        def __get__(self):
            status = self.status
            return np.where((status & 7 != 1) &
                            (status & 7 != 5) &
                            (status & 64 == 0))[0]

    property varIsFree:
        '''
        The index set of variables that are *free*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 0)

    property varIsBasic:
        '''
        The index set of variables that are *basic*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 1)

    property varIsAtUpperBound:
        '''
        The index set of variables that are at their upper bound.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 2)

    property varIsAtLowerBound:
        '''
        The index set of variables that are at their lower bound.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 3)

    property varIsSuperBasic:
        '''
        The index set of variables that are *superbasic*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 4)

    property varIsFixed:
        '''
        The index set of variables that are *fixed*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 == 5)

    property varIsFlagged:
        '''
        The index set of variables that are *flagged*.
        '''
        def __get__(self):
            status = self.status
            return (status & 64 != 0)

    property varNotFree:
        '''
        The index set of variables that are NOT *free*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 0)

    property varNotBasic:
        '''
        The index set of variables that are NOT *basic*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 1)

    property varNotAtUpperBound:
        '''
        The index set of variables that are NOT at their upper bound.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 2)

    property varNotAtLowerBound:
        '''
        The index set of variables that are NOT at their lower bound.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 3)

    property varNotSuperBasic:
        '''
        The index set of variables that are NOT *superbasic*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 4)

    property varNotFixed:
        '''
        The index set of variables that are NOT *fixed*.
        '''
        def __get__(self):
            status = self.status
            return (status & 7 != 5)

    property varNotFlagged:
        '''
        The index set of variables that are NOT flagged.
        '''
        def __get__(self):
            status = self.status
            return (status & 64 == 0)

    property Hessian:
        def __get__(self):
            return self.Hessian

    #############################################
    # get set
    #############################################

    def getRightHandSide(self, np.ndarray[np.double_t, ndim=1] rhs):
        '''
        Take a spare array, ``rhs``, and store the current right-hand-side
        in it.
        '''
        self.CppSelf.getRightHandSide(<double*>rhs.data)

    def getStatusCode(self):
        '''
        Get the probelm status as defined in CLP. Return value could be:

        * -1 - unknown e.g. before solve or if postSolve says not optimal
        * 0 - optimal
        * 1 - primal infeasible
        * 2 - dual infeasible
        * 3 - stopped on iterations or time
        * 4 - stopped due to errors
        * 5 - stopped by event handler (virtual int ClpEventHandler::event())

        '''
        return self.CppSelf.status()

    def getStatusString(self):
        '''
        Return the problem status in string using the code from
        :func:`getStatusCode`
        '''
        return problemStatus[self.getStatusCode()]

    def setColumnLower(self, ind, val):
        '''
        Set the lower bound of variable index ``ind`` to ``val``.
        '''
        self.CppSelf.setColumnLower(ind, val)

    def setColumnUpper(self, ind, val):
        '''
        Set the upper bound of variable index ``ind`` to ``val``.
        '''
        self.CppSelf.setColumnUpper(ind, val)

    def setRowLower(self, ind, val):
        '''
        Set the lower bound of constraint index ``ind`` to ``val``.
        '''
        self.CppSelf.setRowLower(ind, val)

    def setRowUpper(self, ind, val):
        '''
        Set the upper bound of constraint index ``ind`` to ``val``.
        '''
        self.CppSelf.setRowUpper(ind, val)

    def useCustomPrimal(self, customPrimal):
        '''
        Determines if
        :func:`CyLP.python.pivot.PivotPythonBase.isPivotAcceptable`
        should be called just before each pivot is performed (right after the
        entering and leaving variables are obtained.
        '''
        self.CppSelf.useCustomPrimal(customPrimal)

    def getUseCustomPrimal(self):
        '''
        Return the value of ``useCustomPrimal``. See :func:`useCustomPrimal`.

        :rtype: int  :math:`\in \{0, 1\}`
        '''
        return self.CppSelf.getUseCustomPrimal()

    def flagged(self, varInd):
        '''
        Returns ``1`` if variable index ``varInd`` is flagged.

        :rtype: int  :math:`\in \{0, 1\}`
        '''
        return self.CppSelf.flagged(varInd)

    def setFlagged(self, varInd):
        '''
        Set variables index ``varInd`` flagged.
        '''
        self.CppSelf.setFlagged(varInd)

    def currentDualTolerance(self):
        return self.CppSelf.currentDualTolerance()

    def largestDualError(self):
        return self.CppSelf.largestDualError()

    def pivotRow(self):
        '''
        Return the index of the constraint corresponding to the (basic) leaving
        variable.

        :rtype: int
        '''
        return self.CppSelf.pivotRow()

    def setPivotRow(self, v):
        '''
        Set the ``v``\ 'th variable of the basis as the leaving variable.
        '''
        self.CppSelf.setPivotRow(v)

    def sequenceIn(self):
        '''
        Return the index of the entering variable.

        :rtype: int
        '''
        return self.CppSelf.sequenceIn()

    def setSequenceIn(self, v):
        '''
        Set the variable index ``v`` as the entering variable.
        '''
        self.CppSelf.setSequenceIn(v)

    def dualTolerance(self):
        '''
        Return the dual tolerance.

        :rtype: float
        '''
        return self.CppSelf.dualTolerance()

    cdef double* rowLower(self):
        '''
        Return the lower bounds of the constraints as a double*.
        This can be used only in Cython.
        '''
        return self.CppSelf.rowLower()

    cdef double* rowUpper(self):
        '''
        Return the upper bounds of the constraints as a double*.
        This can be used only in Cython.
        '''
        return self.CppSelf.rowUpper()

    cpdef setVariableName(self, varInd, name):
        '''
        Set the name of variable index ``varInd`` to ``name``.

        :arg varInd: variable index
        :type varInd: integer
        :arg name: desired name for the variable
        :type name: string

        '''
        self.CppSelf.setVariableName(varInd, name)

    cdef int* pivotVariable(self):
        '''
        Return the index set of the basic variables.

        :rtype: int*
        '''
        return self.CppSelf.pivotVariable()

    cpdef  getPivotVariable(self):
        '''
        Return the index set of the basic variables.

        :rtype: Numpy array
        '''
        return <object>self.CppSelf.getPivotVariable()

    cpdef getVarStatus(self, int sequence):
        '''
        gets the status of a variable

        * free : 0
        * basic : 1
        * atUpperBound : 2
        * atLowerBound : 3
        * superBasic : 4
        * fixed : 5

        :rtype: int
        '''
        return self.CppSelf.getStatus(sequence)

    def setColumnUpperArray(self, np.ndarray[np.double_t, ndim=1] columnUpper):
        self.CppSelf.setColumnUpperArray(<double*>columnUpper.data)

    def setColumnLowerArray(self, np.ndarray[np.double_t, ndim=1] columnLower):
        self.CppSelf.setColumnLowerArray(<double*>columnLower.data)

    def setRowUpperArray(self, np.ndarray[np.double_t, ndim=1] rowUpper):
        self.CppSelf.setRowUpperArray(<double*>rowUpper.data)

    def setRowLowerArray(self, np.ndarray[np.double_t, ndim=1] rowLower):
        self.CppSelf.setRowLowerArray(<double*>rowLower.data)

    def setObjectiveArray(self, np.ndarray[np.double_t, ndim=1] objective):
        self.CppSelf.setObjectiveArray(<double*>objective.data, len(objective))

    cdef double* primalColumnSolution(self):
        return self.CppSelf.primalColumnSolution()

    cdef double* dualColumnSolution(self):
        return self.CppSelf.dualColumnSolution()

    cdef double* primalRowSolution(self):
        return self.CppSelf.primalRowSolution()

    cdef double* dualRowSolution(self):
        return self.CppSelf.dualRowSolution()

    #############################################
    # CLP Methods
    #############################################

    def initialSolve(self):
        '''
        Run CLP's initialSolve. It does a presolve and uses primal or dual
        Simplex to solve a problem.

        **Usage example**

        >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.initialSolve()
        'optimal'
        >>> round(s.objectiveValue, 4)
        2520.5717

        '''
        return problemStatus[self.CppSelf.initialSolve()]

    def initialPrimalSolve(self):
        '''
        Run CLP's initalPrimalSolve. The same as :func:`initalSolve` but force
        the use of primal Simplex.

        **Usage example**

        >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.initialPrimalSolve()
        'optimal'
        >>> round(s.objectiveValue, 4)
        2520.5717

        '''
        return problemStatus[self.CppSelf.initialPrimalSolve()]

    def initialDualSolve(self):
        '''
        Run CLP's initalPrimalSolve. The same as :func:`initalSolve` but force
        the use of dual Simplex.

        **Usage example**

        >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.initialDualSolve()
        'optimal'
        >>> round(s.objectiveValue, 4)
        2520.5717

        '''
        return problemStatus[self.CppSelf.initialDualSolve()]

    def addConstraint(self, cons):
        '''
        Adds constraints ``cons``  to the problem. Example for the value
        of ``cons`` is ``0 <= A * x <= b`` where ``A`` is a Numpy matrix and
        b is a :py:class:`CyLPArray`.
        '''
        if self.cyLPModel:
            self.cyLPModel.addConstraint(cons)
            self.loadFromCyLPModel(self.cyLPModel)
        else:
            raise Exception('To add a constraint you must set ' \
                            'CyLPSimplex.cyLPModel first.')

    def addConstraint(self, numberInRow,
                    np.ndarray[np.int32_t, ndim=1] columns,
                    np.ndarray[np.double_t, ndim=1] elements,
                    rowLower,
                    rowUpper):
        '''
        Add a constraint to the problem, CLP style. See CLP documentation.
        Not commonly used in CyLP.
        For CyLP modeling tool see :mod:`CyLP.python.modeling.CyLPModel`.
        '''
        # TODO: This makes adding a row real slower,
        # but it is better than a COIN EXCEPTION!
        for c in columns:
            if c >= self.getNumCols():
                raise Exception('CyClpSimplex.pyx:addConstraint: Column ' \
                                'number %d should be less than column ' \
                                'size %d' % (c, self.getNumCols()))
        self.CppSelf.addRow(numberInRow, <int*>columns.data,
                            <double*>elements.data, rowLower, rowUpper)

    def addVariable(self, numberInColumn,
                        np.ndarray[np.int32_t, ndim=1] rows,
                        np.ndarray[np.double_t, ndim=1] elements,
                        columnLower,
                        columnUpper,
                        objective):
        '''
        Add a variable to the problem, CLP style. See CLP documentation.
        For CyLP modeling tool see :mod:`CyLP.python.modeling.CyLPModel`.
        '''
        # TODO: This makes adding a column real slower,
        # but it is better than a COIN EXCEPTION!
        for r in rows:
            if r >= self.getNumRows():
                raise Exception('CyClpSimplex.pyx:addColumn: Row '\
                        'number %d should be less than row size ' \
                        '%d' % (r, self.getNumRows()))
        self.CppSelf.addColumn(numberInColumn, <int*>rows.data,
                <double*> elements.data, columnLower,
                               columnUpper, objective)

    def addVariables(self, number,
                        np.ndarray[np.double_t, ndim=1] columnLower,
                        np.ndarray[np.double_t, ndim=1] columnUpper,
                        np.ndarray[np.double_t, ndim=1] objective,
                        np.ndarray[np.int32_t, ndim=1] columnStarts,
                        np.ndarray[np.int32_t, ndim=1] rows,
                        np.ndarray[np.double_t, ndim=1] elements):
        '''
        Add ``number`` variables at once, CLP style.
        For CyLP modeling tool see :mod:`CyLP.python.modeling.CyLPModel`.
        '''
        self.CppSelf.addColumns(number, <double*>columnLower.data,
                                        <double*>columnUpper.data,
                                        <double*>objective.data,
                                        <int*>columnStarts.data,
                                        <int*>rows.data,
                                        <double*>elements.data)

    def addConstraints(self, number,
                        np.ndarray[np.double_t, ndim=1] rowLower,
                        np.ndarray[np.double_t, ndim=1] rowUpper,
                        np.ndarray[np.int32_t, ndim=1] rowStarts,
                        np.ndarray[np.int32_t, ndim=1] columns,
                        np.ndarray[np.double_t, ndim=1] elements):
        '''
        Add ``number`` constraints at once, CLP style.
        For CyLP modeling tool see :mod:`CyLP.python.modeling.CyLPModel`.
        '''
        self.CppSelf.addRows(number, <double*>rowLower.data,
                                    <double*>rowUpper.data,
                                    <int*>rowStarts.data,
                                    <int*>columns.data,
                                    <double*>elements.data)

    cpdef int readMps(self, char* filename, int keepNames=False,
            int ignoreErrors=False) except *:
        '''
        Read an mps file. See this :ref:`modeling example <modeling-usage>`.
        '''
        name, ext = os.path.splitext(filename)
        if ext not in ['.mps', '.qps']:
            print 'unrecognised extension %s' % ext
            return -1

        if ext == '.mps':
            return self.CppSelf.readMps(filename, keepNames, ignoreErrors)
        else:
            m = CyCoinMpsIO.CyCoinMpsIO()
            ret = m.readMps(filename)
            self.Hessian = m.Hessian
            self.loadProblem(m.matrixByCol, m.variableLower, m.variableUpper,
                             m.objCoefficients,
                             m.constraintLower, m.constraintUpper)
            return ret

    def primal(self, ifValuesPass=0, startFinishOptions=0):
        '''
        Solve the problem using the primal simplex algorithm.
        See this :ref:`usage example <simple-run>`.
        '''
        return problemStatus[self.CppSelf.primal(
                             ifValuesPass, startFinishOptions)]

    def dual(self, ifValuesPass=0, startFinishOptions=0):
        '''
        Runs CLP dual simplex.

        **Usage Example**

        >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.dual()
        'optimal'

        '''
        return problemStatus[self.CppSelf.dual(
                            ifValuesPass, startFinishOptions)]

    def setPerturbation(self, value):
        '''
        Perturb the problem by ``value``.
        '''
        self.CppSelf.setPerturbation(value)

    cdef setPrimalColumnPivotAlgorithm(self, void* choice):
        '''
        Set primal simplex's pivot rule to ``choice``
        This is used when setting a pivot rule in Cython
        '''
        cdef CppClpPrimalColumnPivot* c = <CppClpPrimalColumnPivot*> choice
        self.CppSelf.setPrimalColumnPivotAlgorithm(c)

    def resize(self, newNumberRows, newNumberColumns):
        '''
        Resize the problem. After a call to ``resize`` the problem will have
        ``newNumberRows`` constraints and ``newNumberColumns`` variables.
        '''
        self.CppSelf.resize(newNumberRows, newNumberColumns)

    def getBInvACol(self, col, np.ndarray[np.double_t, ndim=1] cl):
        '''
        Compute :math:`A_B^{-1}A_{col}` and store the result in ``cl``.
        '''
        self.CppSelf.getBInvACol(col, <double*>cl.data)

    def transposeTimesSubset(self, number,
                             np.ndarray[np.int64_t, ndim=1] which,
                             np.ndarray[np.double_t, ndim=1] pi,
                             np.ndarray[np.double_t, ndim=1] y):
        '''
        Compute :math:`y_{which} - pi^{T}A_{which}` where ``which`` is a
        variable index set. Store the result in ``y``.
        '''
        self.CppSelf.transposeTimesSubset(number, <int*>which.data,
                                          <double*>pi.data, <double*>y.data)

    def transposeTimesSubsetAll(self,
                             np.ndarray[np.int64_t, ndim=1] which,
                             np.ndarray[np.double_t, ndim=1] pi,
                             np.ndarray[np.double_t, ndim=1] y):
        '''
        Same as :func:`transposeTimesSubset` but here ``which``
        can also address slack variables.
        '''
        self.CppSelf.transposeTimesSubsetAll(len(which),
                                            <long long int*>which.data,
                                            <double*>pi.data,
                                            <double*>y.data)

    def setInteger(self, arg):
        '''
        if ``arg`` is an integer: mark variable index ``arg`` as integer.
        if ``arg`` is a :class:`CyLPVar` object: mark variable
        ``arg`` as integer. Here is an example of the latter:

        >>> import numpy as np
        >>> from CyLP.cy import CyClpSimplex
        >>> from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray
        >>> model = CyLPModel()
        >>>
        >>> x = model.addVariable('x', 3)
        >>> y = model.addVariable('y', 2)
        >>>
        >>> A = np.matrix([[1., 2., 0],[1., 0, 1.]])
        >>> B = np.matrix([[1., 0, 0], [0, 0, 1.]])
        >>> D = np.matrix([[1., 2.],[0, 1]])
        >>> a = CyLPArray([5, 2.5])
        >>> b = CyLPArray([4.2, 3])
        >>> x_u= CyLPArray([2., 3.5])
        >>>
        >>> model.addConstraint(A*x <= a)
        >>> model.addConstraint(2 <= B * x + D * y <= b)
        >>> model.addConstraint(y >= 0)
        >>> model.addConstraint(1.1 <= x[1:3] <= x_u)
        >>>
        >>> c = CyLPArray([1., -2., 3.])
        >>> model.objective = c * x + 2 * y
        >>>
        >>>
        >>> s = CyClpSimplex(model)
        >>> s.setInteger(x[1:3])
        >>>
        >>> cbcModel = s.getCbcModel()
        >>> cbcModel.branchAndBound()
        >>>
        >>> sol = cbcModel.primalVariableSolution
        >>> (abs(sol -
        ...     np.array([0.5, 2, 2, 0, 0.75]) ) <= 10**-6).all()
        True

        '''

        if isinstance(arg, (int, long)):
            self.CppSelf.setInteger(arg)
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel == None:
                raise Exception('The argument of setInteger can be ' \
                                'a CyLPVar only if the object is built ' \
                                'using a CyLPModel.')
            var = arg
            model = self.cyLPModel
            inds = model.inds
            varName = var.name
            if not inds.hasVar(varName):
                raise Exception('No such variable: %s' % varName)
            x = inds.varIndex[varName]
            if var.parent:
                for i in var.indices:
                    self.CppSelf.setInteger(x[i])
            else:
                for i in xrange(var.dim):
                    self.CppSelf.setInteger(x[i])

    def copyInIntegerInformation(self, np.ndarray[np.uint8_t, ndim=1] colType):
        '''
        Take in a character array containing 0-1 specifying whether or not
        a variable is integer
        '''
        self.CppSelf.copyInIntegerInformation(<char*>colType.data)

    def replaceMatrix(self, CyCoinPackedMatrix matrix, deleteCurrent=False):
        self.CppSelf.replaceMatrix(matrix.CppSelf, deleteCurrent)

    def preSolve(self, feasibilityTolerance=0.0,
                 keepIntegers=0, numberPasses=5,
                 dropNames=0, doRowObjective=0):
        cdef CppIClpSimplex* model = self.CppSelf.preSolve(self.CppSelf,
                                feasibilityTolerance, keepIntegers,
                                numberPasses, dropNames, doRowObjective)
        if model == NULL:
            print "Presolve says problem infeasible."
            return

        self.setCppSelf(model)

    def writeMps(self, filename, formatType=0, numberAcross=2, objSense=0):
        try:
            f = open(filename, 'w')
            f.close()
        except:
            raise Exception('No write access for %s or an intermediate \
                            directory does not exist.' % filename)
        return self.CppSelf.writeMps(filename, formatType, numberAcross,
                                     objSense)
    #############################################
    # Modeling
    #############################################

    def loadFromCyLPModel(self, cyLPModel):
        '''
        Set the coefficient matrix, constraint bounds, and variable
        bounds based on the data in *cyLPModel* which should be and object
        of *CyLPModel* class.

        This method is usually called from CyClpSimplex's constructor.
        But in a case that the CyClpSimplex instance is created before
        we have the CyLPModel we use this method to load the LP,
        for example:

        >>> from CyLP.cy.CyClpSimplex import CyClpSimplex, getModelExample
        >>>
        >>> s = CyClpSimplex()
        >>> model = getModelExample()
        >>> s.loadFromCyLPModel(model)
        >>>
        >>> s.primal()
        'optimal'
        >>> s.primalVariableSolution
        array([ 0.2,  2. ,  1.1,  0. ,  0.9])

        '''

        (mat, constraintLower, constraintUpper,
                    variableLower, variableUpper) = cyLPModel.makeMatrices()

#        coinMat = CyCoinPackedMatrix()
#        coinMat.makeMatrixFromCoo(True, np.array(mat.row, np.int32),
#                                        np.array(mat.col, np.int32),
#                                        np.array(mat.data, np.double))
        coinMat = CyCoinPackedMatrix(True, np.array(mat.row, np.int32),
                                        np.array(mat.col, np.int32),
                                        np.array(mat.data, np.double))

        constraints = cyLPModel.constraints

        #start adding the arrays and the matrix to the problem
        self.resize(len(constraintLower), len(variableLower))

        for i in xrange(len(variableLower)):
            self.setColumnLower(i, variableLower[i])
            self.setColumnUpper(i, variableUpper[i])

        for i in xrange(len(constraintLower)):
            self.setRowLower(i, constraintLower[i])
            self.setRowUpper(i, constraintUpper[i])

        #setting integer informations
        variables = cyLPModel.variables
        curVarInd = 0
        for var in variables:
            if var.isInt:
                for i in xrange(curVarInd, curVarInd + var.dim):
                    self.setInteger(i)
            curVarInd += var.dim

        if cyLPModel.objective != None:
            self.setObjectiveArray(cyLPModel.objective.astype(np.double))
        self.replaceMatrix(coinMat, True)

    #############################################
    # Integer Programming
    #############################################

    def getCbcModel(self):
        '''
        Run initialSolve, return a :class:`CyCbcModel` object that can be
        used to add cuts, run B&B and ...
        '''
        cdef CppICbcModel* model = self.CppSelf.getICbcModel()
        return CyCbcModel().setCppSelf(model)

    #############################################
    # CyLP and Pivoting
    #############################################

    def isPivotAcceptable(self):
        return (<CyPivotPythonBase>
                self.cyPivot).pivotMethodObject.isPivotAcceptable()

    def checkVar(self, i):
        (<CyPivotPythonBase>self.cyPivot).pivotMethodObject.checkVar(i)
        return (<CyPivotPythonBase>self.cyPivot).pivotMethodObject.checkVar(i)

    def setPrimalColumnPivotAlgorithmToWolfe(self):
        '''
        Set primal simplex's pivot rule to the Cython implementation of
        Wolfe's rule used to solve QPs.
        '''
        cdef CyWolfePivot wp = CyWolfePivot()
        self.setPrimalColumnPivotAlgorithm(wp.CppSelf)

    def setPrimalColumnPivotAlgorithmToPE(self):
        '''
        Set primal simplex's pivot rule to the Cython
        implementation of *positive edge*
        '''
        cdef CyPEPivot pe = CyPEPivot()
        self.setPrimalColumnPivotAlgorithm(pe.CppSelf)

    def setPivotMethod(self, pivotMethodObject):
        '''
        Takes a python object and sets it as the primal
        simplex pivot rule. ``pivotObjectMethod`` should
        implement :py:class:`PivotPythonBase`.
        See :ref:`how to use custom Python pivots
        to solve LPs <custom-pivot-usage>`.
        '''
        if not issubclass(pivotMethodObject.__class__, PivotPythonBase):
            raise TypeError('pivotMethodObject should be of a \
                            class derived from PivotPythonBase')

        cdef CyPivotPythonBase p = CyPivotPythonBase(pivotMethodObject)
        self.cyPivot = p
        p.cyModel = self
        self.setPrimalColumnPivotAlgorithm(p.CppSelf)

    cpdef filterVars(self,  inds):
        return <object>self.CppSelf.filterVars(<PyObject*>inds)

    def setObjectiveCoefficient(self, elementIndex, elementValue):
        '''
        Set the objective coefficients using sparse vector elements
        ``elementIndex`` and ``elementValue``.
        '''
        self.CppSelf.setObjectiveCoefficient(elementIndex, elementValue)

    def partialPricing(self, start, end,
                      np.ndarray[np.int32_t, ndim=1] numberWanted):
        '''
        Perform partial pricing from variable ``start`` to variable ``end``.
        Stop when ``numberWanted`` variables good variable checked.
        '''
        return self.CppSelf.partialPrice(start, end, <int*>numberWanted.data)

    def setComplementarityList(self, np.ndarray[np.int32_t, ndim=1] cl):
        self.CppSelf.setComplementarityList(<int*>cl.data)

    cpdef getACol(self, int ncol, CyCoinIndexedVector colArray):
        '''
        Gets column ``ncol`` of ``A`` and store it in ``colArray``.
        '''
        self.CppSelf.getACol(ncol, colArray.CppSelf)

    cpdef vectorTimesB_1(self, CyCoinIndexedVector vec):
        '''
        Compute :math:`vec A_B^{-1}` and store it in ``vec``.
        '''
        self.CppSelf.vectorTimesB_1(vec.CppSelf)

    cdef primalRow(self, CppCoinIndexedVector * rowArray,
                                       CppCoinIndexedVector * rhsArray,
                                       CppCoinIndexedVector * spareArray,
                                       CppCoinIndexedVector * spareArray2,
                                       int valuesPass):
        raise Exception('CyClpPrimalColumnPivotBase.pyx: pivot column ' \
                        'should be implemented.')

    def argWeightedMax(self, arr, arr_ind, w, w_ind):
        return self.CppSelf.argWeightedMax(<PyObject*>arr, <PyObject*>arr_ind,
                                            <PyObject*>w, <PyObject*>w_ind)

#    def getnff(self):
#        status = self.status
#        return np.where((status & 7 != 5) & (status & 64 == 0))[0]
#
#    def getfs(self):
#        status = self.status
#        return np.where((status & 7 == 4) | (status & 7 == 0))[0]

    cdef int* ComplementarityList(self):
        return self.CppSelf.ComplementarityList()

    cpdef getComplementarityList(self):
        return <object>self.CppSelf.getComplementarityList()

    def setComplement(self, var1, var2):
        'sets var1 and var2 to be complements'
        #When you create LP using CoinModel getComplementarityList
        #cannot return with the right size
        #cl = self.getComplementarityList()
        #print var1, var2, len(cl)
        #cl[var1], cl[var2] = var2, var1
        self.CppSelf.setComplement(var1, var2)

    def loadProblemFromCyCoinModel(self, CyCoinModel modelObject, int
                                        tryPlusMinusOne=False):
        return self.CppSelf.loadProblem(modelObject.CppSelf, tryPlusMinusOne)

    def loadProblem(self, CyCoinPackedMatrix matrix,
                 np.ndarray[np.double_t, ndim=1] collb,
                 np.ndarray[np.double_t, ndim=1] colub,
                 np.ndarray[np.double_t, ndim=1] obj,
                 np.ndarray[np.double_t, ndim=1] rowlb,
                 np.ndarray[np.double_t, ndim=1] rowub,
                 np.ndarray[np.double_t, ndim=1] rowObjective=np.array([])):
        cdef double* rd
        if len(rowObjective) == 0:
            rd = NULL
        else:
            rd = <double*> rowObjective.data
        self.CppSelf.loadProblem(matrix.CppSelf, <double*> collb.data,
                                         <double*> colub.data,
                                         <double*> obj.data,
                                         <double*> rowlb.data,
                                         <double*> rowub.data,
                                         <double*> rd)

    def getCoinInfinity(self):
        return self.CppSelf.getCoinInfinity()

#cdef api void CyPostPrimalRow(CppIClpSimplex* s):
#    cl = s.ComplementarityList()
#    pivotRow = s.pivotRow()
#    if pivotRow < 0:
#        return
#    leavingVarIndex = s.pivotVariable()[pivotRow]
#    colInd = s.sequenceIn()
#
#    print 'in: ', colInd
#    print 'out: ', leavingVarIndex
#
#    print 'Basis'
#    for i in xrange(s.getNumRows()):
#        print s.pivotVariable()[i],
#    print
#
#    print 'status : ', s.getStatus(cl[colInd])
#
#    if s.getStatus(cl[colInd]) == 1 and \
#        cl[colInd] != leavingVarIndex:
#
#        print colInd, ' flagged'
#        s.setFlagged(colInd)
#        s.setPivotRow(-3)
#
#cdef api int CyPivotIsAcceptable(CppIClpSimplex* s):
#    cl = s.ComplementarityList()
#    pivotRow = s.pivotRow()
#    if pivotRow < 0:
#        return 1
#
#    leavingVarIndex = s.pivotVariable()[pivotRow]
#    colInd = s.sequenceIn()
#
#    print '____________________\nBasis'
#    for i in xrange(s.getNumRows()):
#        print s.pivotVariable()[i],
#    print
#
#    print 'in: ', colInd
#    print 'out: ', leavingVarIndex
#
#    if s.getStatus(cl[colInd]) == 1 and \
#        cl[colInd] != leavingVarIndex:
#        print colInd, ' flagged'
#        s.setFlagged(colInd)
#        return 0
#
#    return 1


def getModelExample():
    '''
    Return a model example to be used in doctests.
    '''
    import numpy as np
    from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray
    from CyLP.cy import CyClpSimplex

    model = CyLPModel()
    x = model.addVariable('x', 3)
    y = model.addVariable('y', 2)

    A = np.matrix([[1., 2., 0], [1., 0, 1.]])
    B = np.matrix([[1., 0, 0], [0, 0, 1.]])
    D = np.matrix([[1., 2.], [0, 1]])
    a = CyLPArray([5, 2.5])
    b = CyLPArray([4.2, 3])
    x_u= CyLPArray([2., 3.5])

    model.addConstraint(A * x <= a)
    model.addConstraint(2 <= B * x + D * y <= b)
    model.addConstraint(y >= 0)
    model.addConstraint(1.1 <= x[1:3] <= x_u)

    c = CyLPArray([1., -2., 3.])
    model.objective = c * x + 2 * y

    return model


cpdef cydot(CyCoinIndexedVector v1, CyCoinIndexedVector v2):
    return cdot(v1.CppSelf, v2.CppSelf)


def getMpsExample():
    '''
    Return full path to an MPS example file for doctests
    '''
    import os
    import inspect
    curpath = os.path.dirname(inspect.getfile(inspect.currentframe()))
    return os.path.join(curpath, '../input/p0033.mps')


cdef int RunIsPivotAcceptable(void * ptr):
    cdef CyClpSimplex CyWrapper = <CyClpSimplex>(ptr)
    return CyWrapper.isPivotAcceptable()


cdef int RunVarSelCriteria(void * ptr, int varInd):
    cdef CyClpSimplex CyWrapper = <CyClpSimplex>(ptr)
    return CyWrapper.checkVar(varInd)


cdef class VarStatus:
    free = 0
    basic = 1
    atUpperBound = 2
    atLowerBound = 3
    superBasic = 4
    fixed = 5
    status_ = np.array([free,
                        basic,
                        atUpperBound,
                        atLowerBound,
                        superBasic,
                        fixed])
