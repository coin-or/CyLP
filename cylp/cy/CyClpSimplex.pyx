# cython: c_string_type=str, c_string_encoding=ascii
# cython: profile=True
# cython: embedsignature=True

import inspect
import os.path
from itertools import product
import numpy as np
cimport numpy as np
from scipy import sparse
cimport cpython.ref as cpy_ref
from cylp.cy.CyWolfePivot cimport CyWolfePivot
from cylp.cy.CyPEPivot cimport CyPEPivot
from cylp.cy.CyPivotPythonBase cimport CyPivotPythonBase
from cylp.cy.CyDualPivotPythonBase cimport CyDualPivotPythonBase
from cylp.cy cimport CyCoinModel
from cylp.py.utils.sparseUtil import sparseConcat, csc_matrixPlus
from cylp.py.modeling.CyLPModel import CyLPVar, CyLPArray, CyLPSolution
from cylp.py.pivots.PivotPythonBase import PivotPythonBase
from cylp.py.pivots.DualPivotPythonBase import DualPivotPythonBase
from cylp.py.modeling.CyLPModel import CyLPModel
from cylp.cy cimport CyCoinMpsIO

# Initialize numpy
np.import_array()

problemStatus = ['optimal', 'primal infeasible', 'dual infeasible',
                'stopped on iterations or time',
                'stopped due to errors',
                'stopped by event handler (virtual int ' \
                                    'ClpEventHandler::event())']

CLP_variableStatusEnum = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05]
StatusToInt = {'free' : 0, 'basic' : 1, 'atUpperBound' : 2,
                             'atLowerBound' : 3, 'superBasic' : 4,'fixed' : 5}
IntToStatus = ['free', 'basic', 'atUpperBound',
                             'atLowerBound', 'superBasic','fixed']

startFinishOptionsDic = {'x' : 1,  #do not delete work areas
                      'f' : 2, #use old factorization
                      's' : 4} #skip initialization of work areas

cdef class CyClpSimplex:
    '''
    CyClpSimplex is a Cython interface to CLP.
    Not all methods are available but they are being added gradually.

    Its constructor can create an empty object if no argument is provided.
    However, if a :class:`CyLPModel <cylp.py.modeling.CyLPModel>` object is
    given then the resulting ``CyClpSimplex`` object will be build from it.
    For an example of the latter case see
    :mod:`cylp's modeling tool. <cylp.py.modeling.CyLPModel>`

    .. _simple-run:

    **An easy example of how to read and solve an LP**

    >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getMpsExample
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
                                'cylpSimplex constructor. Got %s' %
                                cyLPModel.__class__)

    def __dealloc__(self):
        del self.CppSelf

    cdef setCppSelf(self,  CppIClpSimplex* s):
        del self.CppSelf
        self.CppSelf = s

    #############################################
    # Properties
    #############################################

    property basicVariables:
        def __get__(self):
            cdef np.ndarray[np.int32_t, ndim=1] bv = np.zeros(self.nConstraints, dtype=np.int32)
            self.CppSelf.getBasics(<int*>bv.data)
            return bv

    property rhs:
        def __get__(self):
            r = np.zeros(self.nConstraints, dtype=np.double)
            self.getRightHandSide(r)
            #Py_INCREF(r)
            return r

    property basisInverse:
        def __get__(self):
            b = np.empty((self.nConstraints, self.nConstraints), dtype=np.double)
            cdef np.ndarray[np.double_t, ndim=1] c = np.zeros(self.nConstraints, dtype=np.double)
            for colInd in xrange(self.nConstraints):
                self.getBInvCol(colInd, c)
                b[:, colInd] = c
            return b

    property tableau:
        def __get__(self):
            nAllVars = self.nVariables + self.nConstraints
            t = np.empty((self.nConstraints, nAllVars), dtype=np.double)
            cdef np.ndarray[np.double_t, ndim=1] c = np.zeros(self.nConstraints, dtype=np.double)
            for colInd in xrange(nAllVars):
                self.getBInvACol(colInd, c)
                t[:, colInd] = c
            return t

    property objective:
        '''
        Set the objective function using this property.
        See the :ref:`modeling example <modeling-usage>`.
        '''
        def __set__(self, obj):
            if self.cyLPModel:
                self.cyLPModel.objective = obj
                o = self.cyLPModel.objective

                if not isinstance(o, (np.ndarray)):
                    o = o.toarray()[0]
                self.setObjectiveArray(o.astype(np.double))
                # This old version doesn't work in some versions of Scipy
                # For csr_matrixPlus, o[0,i] is still a matrix, not a number
                # This does work in some versions of SciPy
                # It would probably be OK if csr_matrixPlus didn't override
                # __get_item__ to always cast the result back to csr_matrixPlus
                # I'm not actually sure why the objective is stored as 
                # csr_matrixPlus anyway... seems to not always be true.
                #
                #if isinstance(o, (sparse.coo_matrix,
                #                                sparse.csc_matrix,
                #                                sparse.csr_matrix,
                #                                sparse.lil_matrix)):
                #    for i in xrange(self.nVariables):
                #        self.setObjectiveCoefficient(i, o[0,i])
                    #if not isinstance(o, sparse.coo_matrix):
                    #    o = o.tocoo()
                    #for i, j, v in zip(o.row, o.col, o.data):
                    #    self.setObjectiveCoefficient(j, v)
                #self.setObjectiveArray(
                #       self.cyLPModel.objective.astype(np.double))
            else:
                raise Exception('To set the objective function of ' \
                                'CyClpSimplex set cylpSimplex.cyLPModel ' \
                                'first.')
        def __get__(self):
            return <object>self.CppSelf.getObjective()

    property objectiveCoefficients:
        '''
        An alternative to self.objective, with a more meaningful name
        in a more general context. Sets and gets a numpy array.
        '''
        def __set__(self, coef):
            self.objective = coef

        def __get__(self):
            return self.objective

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

    property coefMatrix:
        '''
        The coefficient matrix. A scipy sparse matrix.
        '''
        def __get__(self):
            mat = self.matrix
            if mat.hasGaps():
                mat.removeGaps()
            return csc_matrixPlus((mat.elements, mat.indices, mat.vectorStarts),
                             shape=(self.nConstraints, self.nVariables))

        def __set__(self, sparseMatrix):
            try:
                m = sparseMatrix.tocoo()
            except:
                raise Exception('coefMatrix must be a scipy sparse matrix.')
            self.matrix = CyCoinPackedMatrix(True, m.row, m.col, m.data)

    property matrix:
        '''
        The coefficient matrix. A CyCoinPackedMatrix.
        '''
        def __get__(self):
            cdef CppCoinPackedMatrix* cppMat = self.CppSelf.getMatrix()
            mat = CyCoinPackedMatrix()
            mat.CppSelf = cppMat
            return mat

        def __set__(self, cyCoinPackedMatrix):
            self.replaceMatrix(cyCoinPackedMatrix, True)

    property constraints:
        '''
        Constraints.
        '''
        def __get__(self):
            if not self.cyLPModel:
                raise Exception('No CyClpSimplex cyLPModel.')
            else:
                return self.cyLPModel.constraints

    property variableNames:
        '''
        variable names
        '''
        def __get__(self):
            return self.getVariableNames()

    property variables:
        '''
        Variables.
        '''
        def __get__(self):
            if not self.cyLPModel:
                raise Exception('No CyClpSimplex cyLPModel.')
            else:
                return self.cyLPModel.variables

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
            ret = <object>self.CppSelf.getPrimalColumnSolution()
            if self.cyLPModel:
                m = self.cyLPModel
                inds = m.inds
                d = {}
                for v in inds.varIndex.keys():
                    d[v] = ret[inds.varIndex[v]]
                    var = m.getVarByName(v)
                    if var.dims:
                        d[v] = CyLPSolution()
                        dimRanges = [range(i) for i in var.dims]
                        for element in product(*dimRanges):
#                            d[v][element] = ret[var.__getitem__(element).indices[0]]
                            d[v][element] = ret[var.fromInd+var[element].indices[0]]
                ret = d
            else:
                names = self.variableNames
                if names:
                    d = CyLPSolution()
                    for i in range(len(names)):
                        d[names[i]] = ret[i]
                    ret = d
            return ret

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

    property cost:
        '''
        Return the cost vector.

        :rtype: Numpy array
        '''
        def __get__(self):
            #if self.cbcModelExists:
            #    return <object>self.cbcModel.getPrimalVariableSolution()
            return <object>self.CppSelf.getCostRegion()

    property dualVariableSolution:
        '''
        Variables' reduced costs

        :rtype: Numpy array
        '''
        def __get__(self):
            ret = <object>self.CppSelf.getDualColumnSolution()
            if self.cyLPModel:
                m = self.cyLPModel
                inds = m.inds
                d = {}
                for v in inds.varIndex.keys():
                    d[v] = ret[inds.varIndex[v]]
                    var = m.getVarByName(v)
                    if var.dims:
                        d[v] = CyLPSolution()
                        dimRanges = [range(i) for i in var.dims]
                        for element in product(*dimRanges):
                            d[v][element] = ret[var.__getitem__(element).indices[0]]
                ret = d
            else:
                names = self.variableNames
                if names:
                    d = CyLPSolution()
                    for i in range(len(names)):
                        d[names[i]] = ret[i]
                    ret = d
            return ret

    property primalConstraintSolution:
        '''
        Slack variables' solution

        :rtype: Numpy array
        '''
        def __get__(self):
            ret = <object>self.CppSelf.getPrimalRowSolution()
            if self.cyLPModel:
                m = self.cyLPModel
                inds = m.inds
                d = {}
                for c in inds.constIndex.keys():
                    d[c] = ret[inds.constIndex[c]]
                ret = d
            else:
                pass
                #names = self.variableNames
                #if names:
                #    d = CyLPSolution()
                #    for i in range(len(names)):
                #        d[names[i]] = ret[i]
                #    ret = d
            return ret


    property dualConstraintSolution:
        '''
        Dual variables' solution

        :rtype: Numpy array
        '''
        def __get__(self):
            ret =  <object>self.CppSelf.getDualRowSolution()
            if self.cyLPModel:
                m = self.cyLPModel
                inds = m.inds
                d = {}
                for c in inds.constIndex.keys():
                    d[c] = ret[inds.constIndex[c]]
                ret = d
            else:
                pass
                #names = self.variableNames
                #if names:
                #    d = CyLPSolution()
                #    for i in range(len(names)):
                #        d[names[i]] = ret[i]
                #    ret = d
            return ret

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

    property objectiveOffset:
        '''
        The constant value in the objective function. A float.
        '''
        def __get__(self):
            return self.CppSelf.objectiveOffset()

        def __set__(self, value):
            self.CppSelf.setObjectiveOffset(value)

    property variablesUpper:
        '''
        Variables upper bounds

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getColUpper()

        def __set__(self, upperArray):
            self.setColumnUpperFirstElements(upperArray)

    property variablesLower:
        '''
        Variables lower bounds

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getColLower()

        def __set__(self, lowerArray):
            self.setColumnLowerFirstElements(lowerArray)

    property constraintsUpper:
        '''
        Constraints upper bounds

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getRowUpper()

        def __set__(self, upperArray):
            self.setRowUpperArray(upperArray)

    property constraintsLower:
        '''
        Constraints lower bounds

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getRowLower()

        def __set__(self, lowerArray):
            self.setRowLowerArray(lowerArray)

    property lower:
        '''
        lower bounds (CLP's lower_)

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getLower()

    property upper:
        '''
        upper bounds (CLP's upper_)

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getUpper()

    property variableScale:
        '''
        Array of variables' scale factors

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getColumnScale()

    property constraintScale:
        '''
        Array of constraints' scale factors

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getRowScale()

    property integerInformation:
        '''
        A binary list of size *nVariables* that specifies whether
        a variable is integer or not. (ClpModel::integerInformation())

        :rtype: Numpy array
        '''
        def __get__(self):
            return <object>self.CppSelf.getIntegerInformation()

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
            return self._Hessian

        def __set__(self, mat):
            m = None
            try:
                m = mat.tocoo()
            except:
                raise Exception('Hessian can be set to a matrix that ' \
                                            'implements *tocoo* method')
            if m:
                coinMat = CyCoinPackedMatrix(True, m.row, m.col, m.data)
                n = self.nVariables
                if coinMat.majorDim < n:
                    for i in xrange(n - coinMat.majorDim):
                        coinMat.appendCol()
                if coinMat.minorDim < n:
                    for i in xrange(n - coinMat.majorDim):
                        coinMat.appendRow()
                self._Hessian = m
            self.loadQuadraticObjective(coinMat)

    property dualTolerance:
        def __get__(self):
            return self.CppSelf.dualTolerance()

        def __set__(self, value):
           self.CppSelf.setDualTolerance(value)

    property primalTolerance:
        def __get__(self):
            return self.CppSelf.primalTolerance()

        def __set__(self, value):
           self.CppSelf.setPrimalTolerance(value)

    property maxNumIteration:
        def __get__(self):
            return self.CppSelf.maximumIterations()

        def __set__(self, value):
           self.CppSelf.setMaxNumIteration(value)

    property logLevel:
        def __get__(self):
            return self.CppSelf.logLevel()

        def __set__(self, value):
            self.CppSelf.setLogLevel(value)

    property automaticScaling:
        def __get__(self):
            return self.CppSelf.automaticScaling()

        def __set__(self, value):
            self.CppSelf.setAutomaticScaling(value)

    property scaling:
        def __get__(self):
            return self.CppSelf.scalingFlag()
        def __set__(self, mode):
            self.CppSelf.scaling(mode)

    property infeasibilityCost:
        def __get__(self):
            return self.CppSelf.infeasibilityCost()
        def __set__(self, value):
            self.CppSelf.setInfeasibilityCost(value)


    property numberPrimalInfeasibilities:
        def __get__(self):
            return self.CppSelf.numberPrimalInfeasibilities()

    property optimizationDirection:
        def __get__(self):
            return ['ignore', 'min', 'max'][int(self.CppSelf.optimizationDirection())]
        def __set__(self, value):
            self.CppSelf.setOptimizationDirection({'ignore':0., 'min':1., 'max':-1.}[value])

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
        :func:`cylp.python.pivot.PivotPythonBase.isPivotAcceptable`
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

##    def currentDualTolerance(self):
##        return self.CppSelf.currentDualTolerance()
##
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

##    def dualTolerance(self):
##        '''
##        Return the dual tolerance.
##
##        :rtype: float
##        '''
##        return self.CppSelf.dualTolerance()

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

    def getVariableNames(self):
        '''
        Return the variable name. (e.g. that was set in the mps file)
        '''
        cdef vector[string] names = self.CppSelf.getVariableNames()
        ret = []
        for i in range(names.size()):
            ret.append(names[i].c_str())
        return ret

    cpdef setVariableName(self, varInd, name):
        '''
        Set the name of variable index ``varInd`` to ``name``.

        :arg varInd: variable index
        :type varInd: integer
        :arg name: desired name for the variable
        :type name: string

        '''
        self.CppSelf.setVariableName(varInd, name)

    cpdef setConstraintName(self, constInd, name):
        '''
        Set the name of constraint index ``constInd`` to ``name``.

        :arg constInd: constraint index
        :type constInd: integer
        :arg name: desired name for the constraint
        :type name: string

        '''
        self.CppSelf.setConstraintName(constInd, name)

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

    cpdef CLP_getVarStatus(self, int sequence):
        '''
        get the status of a variable

        * free : 0
        * basic : 1
        * atUpperBound : 2
        * atLowerBound : 3
        * superBasic : 4
        * fixed : 5

        :rtype: int
        '''
        return self.CppSelf.getStatus(sequence)

    cpdef CLP_setVarStatus(self, int sequence, int status):
        '''
        set the status of a variable

        * free : 0
        * basic : 1
        * atUpperBound : 2
        * atLowerBound : 3
        * superBasic : 4
        * fixed : 5

        '''
        self.CppSelf.setStatus(sequence, CLP_variableStatusEnum[status])

    def setVariableStatus(self, arg, status):
        '''
        Set the status of a variable.

        :arg arg: Specifies the variable to change (a CyLPVar, or an index)
        :type status: CyLPVar, int
        :arg status: 'basic', 'atUpperBound', 'atLowerBound', 'superBasic', 'fixed'
        :type status: string


        Example:

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex
        >>> s = CyClpSimplex()
        >>> x = s.addVariable('x', 4)
        >>> # Using CyLPVars:
        >>> s.setVariableStatus(x[1:3], 'basic')
        >>> s.getVariableStatus(x[1])
        'basic'
        >>> # Using a variable index directly
        >>> s.setVariableStatus(1, 'atLowerBound')
        >>> s.getVariableStatus(x[1])
        'atLowerBound'

        '''
        status = CLP_variableStatusEnum[StatusToInt[status]]
        if isinstance(arg, (int, long)):
            self.CppSelf.setStatus(arg, status)
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
                raise Exception('The argument of setVarStatus can be ' \
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
                    self.CppSelf.setStatus(x[i], status)
            else:
                for i in xrange(var.dim):
                    self.CppSelf.setStatus(x[i], status)

    def getVariableStatus(self, arg):
        '''
        Get the status of a variable.
        '''
        if isinstance(arg, (int, long)):
            return IntToStatus[self.CppSelf.getStatus(arg)]
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
                raise Exception('The argument of getVarStatus can be ' \
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
                s = np.array([IntToStatus[
                            self.CppSelf.getStatus(x[i])]
                            for i in var.indices])
            else:
                s = np.array([IntToStatus[
                            self.CppSelf.getStatus(x[i])]
                            for i in xrange(var.dim)])
            if len(s) == 1:
                return s[0]
            return s

    def setConstraintStatus(self, arg, status):
        '''
        Set the status of a constraint.

        :arg arg: Specifies the constraint to change (name or index)
        :type status: string,int
        :arg status: 'basic', 'atUpperBound', 'atLowerBound', 'superBasic', 'fixed'
        :type status: string

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex
        >>> s = CyClpSimplex()
        >>> x = s.addVariable('x', 4)
        >>> s.addConstraint(0 <= x[0] + x[1] <= 1, 'const1')
        >>> # Using constraint name:
        >>> s.setConstraintStatus('const1', 'atUpperBound')
        >>> s.getConstraintStatus('const1')
        'atUpperBound'
        >>> # Using constraint index directly
        >>> s.setConstraintStatus(0, 'atLowerBound')
        >>> s.getConstraintStatus('const1')
        'atLowerBound'
        '''
        status = CLP_variableStatusEnum[StatusToInt[status]]
        if isinstance(arg, (int, long)):
            arg += self.nVariables
            self.CppSelf.setStatus(arg, status)
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
                raise Exception('The argument of setVarStatus can be ' \
                                'a CyLPVar only if the object is built ' \
                                'using a CyLPModel.')
            model = self.cyLPModel
            inds = model.inds
            constName = arg
            if not inds.hasConst(constName):
                raise Exception('No such constraint: %s' % constName)
            c = inds.constIndex[constName]
            cInds = c + self.nVariables
            for i in xrange(len(cInds)):
                self.CppSelf.setStatus(cInds[i], status)

    def getConstraintStatus(self, arg):
        '''
        Get the status of a constraint.
        '''
        if isinstance(arg, (int, long)):
            arg += self.nVariables
            return IntToStatus[self.CppSelf.getStatus(arg)]
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
                raise Exception('The argument of setVarStatus can be ' \
                                'a CyLPVar only if the object is built ' \
                                'using a CyLPModel.')
            model = self.cyLPModel
            inds = model.inds
            constName = arg
            if not inds.hasConst(constName):
                raise Exception('No such constraint: %s' % constName)
            c = inds.constIndex[constName]
            cInds = c + self.nVariables
            s = np.array([IntToStatus[
                            self.CppSelf.getStatus(cInds[i])]
                            for i in xrange(len(cInds))])
            if len(s) == 1:
                return s[0]
            return s


    def setColumnUpperArray(self, np.ndarray[np.double_t, ndim=1] columnUpper):
        '''
        columnUpper should have n+m elements. The method only does
        a pointer assignment. If you only want to set the first n
        elements use setColumnUpperFirstElements().
        '''
        self.CppSelf.setColumnUpperArray(<double*>columnUpper.data)

    def setColumnUpperFirstElements(self, np.ndarray[np.double_t, ndim=1] columnUpper):
        '''
        Run a loop in C++ and set the first n elements of variables' upperbounds
        '''
        self.CppSelf.setColumnUpperFirstElements(len(columnUpper), <double*>columnUpper.data)

    def setColumnLowerArray(self, np.ndarray[np.double_t, ndim=1] columnLower):
        '''
        columnLower should have n+m elements. The method only does
        a pointer assignment. If you only want to set the first n
        elements use setColumnLowerFirstElements().
        '''
        self.CppSelf.setColumnLowerArray(<double*>columnLower.data)

    def setColumnLowerFirstElements(self, np.ndarray[np.double_t, ndim=1] columnLower):
        '''
        Run a loop in C++ and set the first n elements of variables' lowerbounds
        '''
        self.CppSelf.setColumnLowerFirstElements(len(columnLower), <double*>columnLower.data)

    def setColumnLowerSubset(self, np.ndarray[np.int32_t, ndim=1] indicesOfIndices,
                                   np.ndarray[np.int32_t, ndim=1] indices,
                                   np.ndarray[np.double_t, ndim=1] columnLower):
        '''
        This method is defined for a very specific purpose.
        It's only to be used to speed up self.addConstraint()
        '''
        self.CppSelf.setColumnLowerSubset(len(indicesOfIndices), <int*> indicesOfIndices.data,
                                          <int*> indices.data,
                                          <double*>columnLower.data)

    def setColumnUpperSubset(self, np.ndarray[np.int32_t, ndim=1] indicesOfIndices,
                                   np.ndarray[np.int32_t, ndim=1] indices,
                                   np.ndarray[np.double_t, ndim=1] columnUpper):
        '''
        This method is defined for a very specific purpose.
        It's only to be used to speed up self.addConstraint()
        '''
        self.CppSelf.setColumnUpperSubset(len(indicesOfIndices), <int*> indicesOfIndices.data,
                                          <int*> indices.data,
                                          <double*>columnUpper.data)

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

    def CLP_dualConstraintSolution(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.nConstraints
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_DOUBLE, <void*> self.dualRowSolution())
        return ndarray

    #############################################
    # CLP Methods
    #############################################

    def initialSolve(self, presolve='on'):
        '''
        Run CLP's initialSolve. It does a presolve and uses primal or dual
        Simplex to solve a problem.

        **Usage example**

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.initialSolve()
        'optimal'
        >>> round(s.objectiveValue, 4)
        2520.5717

        '''
        presolve = 0 if presolve == 'on' and self.Hessian is None else 1
        return problemStatus[self.CppSelf.initialSolve(presolve)]

    def initialPrimalSolve(self):
        '''
        Run CLP's initalPrimalSolve. The same as :func:`initalSolve` but force
        the use of primal Simplex.

        **Usage example**

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getMpsExample
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

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getMpsExample
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

    def __iadd__(self, cons):
        self.addConstraint(cons)
        return self

    def addConstraint(self, cons, name='', addMpsNames=True):
        '''
        Adds constraints ``cons``  to the problem. Example for the value
        of ``cons`` is ``0 <= A * x <= b`` where ``A`` is a Numpy matrix and
        b is a :py:class:`CyLPArray`.
        '''
        if self.cyLPModel:
            m = self.cyLPModel
            nVarsBefore = m.nVars
            nConsBefore = m.nCons
            c = m.addConstraint(cons, name, addMpsNames)

            # If the dimension is changing, load from scratch
            if nConsBefore == 0 or m.nVars - nVarsBefore != 0:
                self.loadFromCyLPModel(self.cyLPModel)

            # If the constraint to be added is just a variable range
            elif c.isRange:
                var = c.variables[0]
                dim = var.parentDim
                varinds = m.inds.varIndex[var.name]

                lb = var.parent.lower if var.parent else var.lower
                ub = var.parent.upper if var.parent else var.upper

                #if len(var.indices != 0 :
                self.setColumnLowerSubset(var.indices,
                                              varinds,
                                              lb)
                self.setColumnUpperSubset(var.indices,
                                              varinds,
                                              ub)
                #for i in var.indices:
                #    self.setColumnLower(varinds[i], lb[i])
                #    self.setColumnUpper(varinds[i], ub[i])

            # If the constraint is a "real" constraint, but no
            # dimension changes required
            else:
                mainCoef = None
                for varName in m.varNames:
                    dim = m.pvdims[varName]
                    coef = sparse.coo_matrix((c.nRows, dim))
                    keys = [k for k in c.varCoefs.keys() if k.name == varName]
                    for var in keys:
                        coef = coef + c.varCoefs[var]
                    mainCoef = sparseConcat(mainCoef, coef, 'h')

                self.addConstraints(c.nRows,
                        c.lower, c.upper, mainCoef.indptr,
                        mainCoef.indices, mainCoef.data)
        else:
            raise Exception('To add a constraint you must set ' \
                            'cylpSimplex.cyLPModel first.')

    def removeConstraint(self, name):
        '''
        Removes constraint named ``name`` from the problem.
        '''
        if self.cyLPModel:
            indsOfRemovedConstriants = self.cyLPModel.removeConstraint(name)
            self.CLP_deleteConstraints(indsOfRemovedConstriants)
            #self.loadFromCyLPModel(self.cyLPModel)
        else:
            raise Exception('To remove a constraint you must set ' \
                            'cylpSimplex.cyLPModel first.')

    def addVariable(self, varname, dim, isInt=False):
        '''
        Add variable ``var`` to the problem.
        '''
        if not self.cyLPModel:
            self.cyLPModel = CyLPModel()
        var = self.cyLPModel.addVariable(varname, dim, isInt)
        self.loadFromCyLPModel(self.cyLPModel)
        return var
        #else:
        #    raise Exception('To add a variable you must set ' \
        #                    'cylpSimplex.cyLPModel first.')

    def removeVariable(self, name):
        '''
        Removes variable named ``name`` from the problem.
        '''
        if self.cyLPModel:
            self.cyLPModel.removeVariable(name)
            self.loadFromCyLPModel(self.cyLPModel)
        else:
            raise Exception('To remove a variable you must set ' \
                            'cylpSimplex.cyLPModel first.')

    def getVarByName(self, name):
        if not self.cyLPModel:
            raise Exception('No cylpSimplex.cyLPModel is set.')
        return self.cyLPModel.getVarByName(name)

    def getVarNameByIndex(self, ind):
        if not self.cyLPModel:
            raise Exception('No cylpSimplex.cyLPModel is set.')
        return self.cyLPModel.inds.reverseVarSearch(ind)

    def CLP_addConstraint(self, numberInRow,
                    np.ndarray[np.int32_t, ndim=1] columns,
                    np.ndarray[np.double_t, ndim=1] elements,
                    rowLower,
                    rowUpper):
        '''
        Add a constraint to the problem, CLP style. See CLP documentation.
        Not commonly used in cylp.
        For cylp modeling tool see :mod:`cylp.python.modeling.CyLPModel`.
        '''
        # TODO: This makes adding a row real slower,
        # but it is better than a COIN EXCEPTION!
        if (columns >= self.nVariables).any():
            raise Exception('CyClpSimplex.pyx:addConstraint: Column ' \
                    'index out of range (number of columns: ' \
                                '%d)' % (self.nVariables))
        self.CppSelf.addRow(numberInRow, <int*>columns.data,
                            <double*>elements.data, rowLower, rowUpper)

    def CLP_deleteConstraints(self, np.ndarray[np.int32_t, ndim=1] which):
        '''
        Delete constraints indexed by ``which`` from the LP.
        '''
        if (which >= self.nConstraints).any():
            raise Exception('CyClpSimplex.pyx:deleteConstraints: Constraint ' \
                    'index out of range (number of constraints: ' \
                                '%d)' % (self.nConstraints))
        self.CppSelf.deleteRows(len(which), <int*>which.data)

    def CLP_deleteVariables(self, np.ndarray[np.int32_t, ndim=1] which):
        '''
        Delete variables indexed by ``which`` from the LP.
        '''
        if (which >= self.nVariables).any():
            raise Exception('CyClpSimplex.pyx:deleteVariables: variable ' \
                    'index out of range (number of variables: ' \
                                '%d)' % (self.nVariables))
        self.CppSelf.deleteColumns(len(which), <int*>which.data)

    def CLP_addVariable(self, numberInColumn,
                        np.ndarray[np.int32_t, ndim=1] rows,
                        np.ndarray[np.double_t, ndim=1] elements,
                        columnLower,
                        columnUpper,
                        objective):
        '''
        Add a variable to the problem, CLP style. See CLP documentation.
        For cylp modeling tool see :mod:`cylp.python.modeling.CyLPModel`.
        '''
        # TODO: This makes adding a column real slower,
        # but it is better than a COIN EXCEPTION!
        if (rows >= self.nConstraints).any():
            raise Exception('CyClpSimplex.pyx:addColumn: Row '\
                    'index out of range (number of rows:  ' \
                        '%d)' % (self.nConstraints))
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
        For cylp modeling tool see :mod:`cylp.python.modeling.CyLPModel`.
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
        For cylp modeling tool see :mod:`cylp.python.modeling.CyLPModel`.
        '''
        self.CppSelf.addRows(number, <double*>rowLower.data,
                                    <double*>rowUpper.data,
                                    <int*>rowStarts.data,
                                    <int*>columns.data,
                                    <double*>elements.data)

    cpdef int readMps(self, filename, int keepNames=False,
            int ignoreErrors=False) except *:
        '''
        Read an mps file. See this :ref:`modeling example <modeling-usage>`.
        '''
        filename = filename.encode('ascii')
        name, ext = os.path.splitext(filename)
        if ext not in [b'.mps', b'.qps']:
            print('unrecognised extension %s' % ext)
            return -1

        if ext == b'.mps':
            return self.CppSelf.readMps(filename, keepNames, ignoreErrors)
        else:
            m = CyCoinMpsIO.CyCoinMpsIO()
            ret = m.readMps(filename)
            self._Hessian = m.Hessian

            # Since CyCoinMpsIO.readMps seems to be different from ClpModle.readMps
            # for the moment we read the problem again
            # FIXME: should be fixed
            #self.loadProblem(m.matrixByCol, m.variableLower, m.variableUpper,
            #                 m.objCoefficients,
            #                 m.constraintLower, m.constraintUpper)
            #return ret

            return self.CppSelf.readMps(filename, keepNames, ignoreErrors)


    def extractCyLPModel(self, fileName, keepNames=False, ignoreErrors=False):
        if self.readMps(fileName, keepNames, ignoreErrors) != 0:
            return None
        m = CyLPModel()

        x = m.addVariable('x', self.nVariables)

        # Copy is crucial. Memory space should be different than
        # that of Clp. Else, a resize will ruin these.
        c_up = CyLPArray(self.constraintsUpper).copy()
        c_low = CyLPArray(self.constraintsLower).copy()

        mat = self.matrix
        C = csc_matrixPlus((mat.elements, mat.indices, mat.vectorStarts),
                             shape=(self.nConstraints, self.nVariables))

        m += c_low <= C * x <= c_up

        x_up = CyLPArray(self.variablesUpper).copy()
        x_low = CyLPArray(self.variablesLower).copy()

        m += x_low <= x <= x_up

        m.objective = self.objective

        self.cyLPModel = m
        return m

    def _extractStartFinish(self, startFinishOptions):
        if isinstance(startFinishOptions, int):
            sf = startFinishOptions
        else:
            sf = 0
            for option in startFinishOptions:
                sf = sf | startFinishOptionsDic[option]
        return sf

    def primal(self, ifValuesPass=0, startFinishOptions=0, presolve=False):
        '''
        Solve the problem using the primal simplex algorithm.
        See this :ref:`usage example <simple-run>`.

        startFinishOptions is a string containing one or
        more of the following characters:
        'x': do not delete work areas
        'f': use old factorization if possible
        's': skip initialization of work areas
        So one might call ``self.primal(startFinishOptions='sx')``
        '''
        sf = self._extractStartFinish(startFinishOptions)
        if presolve:
            return self.primalWithPresolve()
        else:
            return problemStatus[self.CppSelf.primal(
                             ifValuesPass, sf)]

    def dual(self, ifValuesPass=0, startFinishOptions=0, presolve=False):
        '''
        Runs CLP dual simplex.

        **Usage Example**

        >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getMpsExample
        >>> s = CyClpSimplex()
        >>> f = getMpsExample()
        >>> s.readMps(f)
        0
        >>> s.dual()
        'optimal'

        '''
        sf = self._extractStartFinish(startFinishOptions)
        if presolve:
            return self.dualWithPresolve()
        else:
            return problemStatus[self.CppSelf.dual(
                            ifValuesPass, sf)]

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

    cdef setDualRowPivotAlgorithm(self, void* choice):
        '''
        Set dual simplex's pivot rule to ``choice``
        This is used when setting a pivot rule in Cython
        '''
        cdef CppClpDualRowPivot* c = <CppClpDualRowPivot*> choice
        self.CppSelf.setDualRowPivotAlgorithm(c)

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

    def getBInvCol(self, col, np.ndarray[np.double_t, ndim=1] cl):
        '''
        Return :math:`A_B^{-1}_{col}` and store the result in ``cl``.
        '''
        self.CppSelf.getBInvCol(col, <double*>cl.data)

    def transposeTimes(self, scalar, CyCoinIndexedVector x,
                       CyCoinIndexedVector y, CyCoinIndexedVector z):
        '''
        Compute :math:`x * scalar * A + y` and store the result in ``z``.
        '''
        self.CppSelf.transposeTimes(self.CppSelf, scalar,
                                    x.CppSelf, y.CppSelf, z.CppSelf)

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

    def isInteger(self, ind):
        '''
        Returns True if the variable index ``ind`` is integer.
        '''
        return self.CppSelf.isInteger(ind)

    def setInteger(self, arg):
        '''
        if ``arg`` is an integer: mark variable index ``arg`` as integer.
        if ``arg`` is a :class:`CyLPVar` object: mark variable
        ``arg`` as integer. Here is an example of the latter:

        >>> import numpy as np
        >>> from cylp.cy import CyClpSimplex
        >>> from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
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
        >>> model += A*x <= a
        >>> model += 2 <= B * x + D * y <= b
        >>> model += y >= 0
        >>> model += 1.1 <= x[1:3] <= x_u
        >>>
        >>> c = CyLPArray([1., -2., 3.])
        >>> model.objective = c * x + 2 * y.sum()
        >>>
        >>>
        >>> s = CyClpSimplex(model)
        >>> s.setInteger(x[1:3])
        >>>
        >>> cbcModel = s.getCbcModel()
        >>> cbcModel.solve()
        0
        >>> print(cbcModel.status)
        'solution'
        >>>
        >>> sol_x = cbcModel.primalVariableSolution['x']
        >>> (abs(sol_x -
        ...     np.array([0.5, 2, 2]) ) <= 10**-6).all()
        True
        >>> sol_y = cbcModel.primalVariableSolution['y']
        >>> (abs(sol_y -
        ...     np.array([0, 0.75]) ) <= 10**-6).all()
        True

        '''

        if isinstance(arg, (int, long)):
            self.CppSelf.setInteger(arg)
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
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

    def loadQuadraticObjective(self, CyCoinPackedMatrix matrix):
        self.CppSelf.loadQuadraticObjective(matrix.CppSelf)

    def preSolve(self, feasibilityTolerance=0.0,
                 keepIntegers=0, numberPasses=5,
                 dropNames=0, doRowObjective=0):
        cdef CppIClpSimplex* model = self.CppSelf.preSolve(self.CppSelf,
                                feasibilityTolerance, keepIntegers,
                                numberPasses, dropNames, doRowObjective)
        s = CyClpSimplex()
        if model == NULL:
            print("Presolve says problem infeasible.")
            return s

        s.setCppSelf(model)
        return s
        #self.setCppSelf(model)

    def postSolve(self, updateStatus=True):
        self.CppSelf.postSolve(updateStatus)

    def dualWithPresolve(self, feasibilityTolerance=0.0,
                 keepIntegers=0, numberPasses=5,
                 dropNames=0, doRowObjective=0):
        ret = self.CppSelf.dualWithPresolve(self.CppSelf,
                                feasibilityTolerance, keepIntegers,
                                numberPasses, dropNames, doRowObjective)
        if ret == -2000:
            print("Presolve says problem infeasible.")
            return -2000

        return problemStatus[ret]

    def primalWithPresolve(self, feasibilityTolerance=0.0,
                 keepIntegers=0, numberPasses=5,
                 dropNames=0, doRowObjective=0):
        ret = self.CppSelf.primalWithPresolve(self.CppSelf,
                                feasibilityTolerance, keepIntegers,
                                numberPasses, dropNames, doRowObjective)
        if ret == -2000:
            print("Presolve says problem infeasible.")
            return -2000

        return problemStatus[ret]

    def writeMps(self, filename, formatType=0, numberAcross=2, objSense=0):
        try:
            f = open(filename, 'w')
            f.close()
        except:
            raise Exception('No write access for %s or an intermediate \
                            directory does not exist.' % filename)

        m = self.cyLPModel
        if m:
            inds = m.inds
            for var in m.variables:
                varinds = inds.varIndex[var.name]
                for i in xrange(var.dim):
                    self.setVariableName(varinds[i], var.mpsNames[i])

            for con in m.constraints:
                coninds = inds.constIndex[con.name]
                for i in xrange(con.nRows):
                    self.setConstraintName(coninds[i], con.mpsNames[i])
        return self.CppSelf.writeMps(filename, formatType, numberAcross,
                                     objSense)

    def writeLp(self, filename, extension="", epsilon=10**-5, numberAcross=10,
                        decimals=5, objSense=0.0, useRowNames=1):
        try:
            f = open(filename, 'w')
            f.close()
        except:
            raise Exception('No write access for %s or an intermediate \
                            directory does not exist.' % filename)

        m = self.cyLPModel
        if m:
            inds = m.inds
            for var in m.variables:
                varinds = inds.varIndex[var.name]
                for i in xrange(var.dim):
                    self.setVariableName(varinds[i], var.mpsNames[i])

            for con in m.constraints:
                coninds = inds.constIndex[con.name]
                for i in xrange(con.nRows):
                    self.setConstraintName(coninds[i], con.mpsNames[i])
        self.CppSelf.writeLp(filename, extension, epsilon, numberAcross, decimals, objSense, useRowNames)

    def readLp(self, char *filename, epsilon=10**-5):
        return self.CppSelf.readLp(filename, epsilon)

    def updateColumnFT(self, CyCoinIndexedVector spare, CyCoinIndexedVector updatedColumn):
        return self.CppSelf.updateColumnFT(spare.CppSelf, updatedColumn.CppSelf)

    def updateColumnTranspose(self, CyCoinIndexedVector regionSparse1,
                                    CyCoinIndexedVector regionSparse2):
        return self.CppSelf.updateColumnTranspose(regionSparse1.CppSelf, regionSparse2.CppSelf)

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

        >>> import numpy as np
        >>> from cylp.cy.CyClpSimplex import CyClpSimplex, getModelExample
        >>>
        >>> s = CyClpSimplex()
        >>> model = getModelExample()
        >>> s.loadFromCyLPModel(model)
        >>>
        >>> s.primal()
        'optimal'
        >>> sol_x = s.primalVariableSolution['x']
        >>> (abs(sol_x -
        ...     np.array([0.2, 2, 1.1]) ) <= 10**-6).all()
        True

        '''
        self.cyLPModel = cyLPModel
        (mat, constraintLower, constraintUpper,
                    variableLower, variableUpper) = cyLPModel.makeMatrices()

        n = len(variableLower)
        m = len(constraintLower)
        if n == 0:# or m == 0:
            return

        self.resize(m, n)
        if mat is not None:
            if not isinstance(mat, sparse.coo_matrix):
                mat = mat.tocoo()

            coinMat = CyCoinPackedMatrix(True, np.array(mat.row, np.int32),
                                        np.array(mat.col, np.int32),
                                        np.array(mat.data, np.double))
        else:
            coinMat = CyCoinPackedMatrix(True, np.array([], np.int32),
                                        np.array([], np.int32),
                                        np.array([], np.double))
        self.replaceMatrix(coinMat, True)

        #start adding the arrays and the matrix to the problem

        for i in xrange(n):
            self.setColumnLower(i, variableLower[i])
            self.setColumnUpper(i, variableUpper[i])

        for i in xrange(m):
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


        if cyLPModel.objective is not None:
            self.objective = cyLPModel.objective

    def evaluateAt(self, x0):
        '''
        Evaluate the objective function at x0
        '''
        if self.Hessian is not None:
            return (np.dot(self.objectiveCoefficients, x0) +
                    0.5 * np.dot(x0, self.Hessian.dot(x0)) - self.objectiveOffset)
        else:
            return np.dot(self.objectiveCoefficients, x0) - self.objectiveOffset

    def gradientAt(self, x0):
        if self.Hessian is not None:
            return self.objectiveCoefficients + self.Hessian * x0
        else:
            return self.objectiveCoefficients


    #############################################
    # Integer Programming
    #############################################

    def getCbcModel(self):
        '''
        Run initialSolve, return a :class:`CyCbcModel` object that can be
        used to add cuts, run B&B and ...
        '''
        cdef CppICbcModel* model = self.CppSelf.getICbcModel()
        cm =  CyCbcModel()
        cm.setCppSelf(model)
        cm.setClpModel(self)
        if self.cyLPModel:
            cm.cyLPModel = self.cyLPModel
        return cm

    #############################################
    # cylp and Pivoting
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

    def setDualPivotMethod(self, dualPivotMethodObject):
        '''
        Takes a python object and sets it as the dual
        pivot rule. ``dualPivotObjectMethod`` should
        implement :py:class:`DualPivotPythonBase`.
        See :ref:`how to use custom dual Python pivots
        to solve LPs <custom-dual-pivot-usage>`.       '''
        if not issubclass(dualPivotMethodObject.__class__, DualPivotPythonBase):
            raise TypeError('dualPivotMethodObject should be of a \
                            class derived from DualPivotPythonBase')

        cdef CyDualPivotPythonBase p = CyDualPivotPythonBase(dualPivotMethodObject)
        self.cyDualPivot = p
        p.cyModel = self
        self.setDualRowPivotAlgorithm(p.CppSelf)


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
        '''
        Set ``var1`` as the complementary variable of ``var2``. These
        arguments may be integers signifying indices, or CyLPVars.
        '''

        if isinstance(var1, (int, long)) and isinstance(var2, (int, long)) :
           self.CppSelf.setComplement(var1, var2)
        elif True:  # isinstance(arg, CyLPVar):
            if self.cyLPModel is None:
                raise Exception('The argument of setInteger can be ' \
                                'a CyLPVar only if the object is built ' \
                                'using a CyLPModel.')
            if var1.dim != var2.dim:
                raise Exception('Variables should have the same  ' \
                                'dimensions to be complements.' \
                                ' Got %s: %g and %s: %g' %
                                (var1.name, var1.dim, var2.name, var2.dim))

            model = self.cyLPModel
            inds = model.inds
            vn1 = var1.name
            vn2 = var2.name

            if not inds.hasVar(vn1):
                raise Exception('No such variable: %s' % vn1)
            x1 = inds.varIndex[vn1]
            if not inds.hasVar(vn2):
                raise Exception('No such variable: %s' % vn2)
            x2 = inds.varIndex[vn2]

            for i in xrange(var1.dim):
                self.CppSelf.setComplement(x1[i], x2[i])

#    def setComplement(self, var1, var2):
#        'sets var1 and var2 to be complements'
#        #When you create LP using CoinModel getComplementarityList
#        #cannot return with the right size
#        #cl = self.getComplementarityList()
#        #print(var1, var2, len(cl))
#        #cl[var1], cl[var2] = var2, var1
#        self.CppSelf.setComplement(var1, var2)

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

    #############################################
    # Osi
    #############################################

    def setBasisStatus(self, np.ndarray[np.int32_t, ndim=1] cstat,
                             np.ndarray[np.int32_t, ndim=1] rstat):
        self.CppSelf.setBasisStatus(<int*>cstat.data, <int*>rstat.data)

    def getBasisStatus(self):
        cdef np.ndarray[np.int32_t, ndim=1] cstat = \
                                np.zeros(self.nVariables, dtype='int32')
        cdef np.ndarray[np.int32_t, ndim=1] rstat = \
                                np.zeros(self.nConstraints, dtype='int32')
        self.CppSelf.getBasisStatus(<int*>cstat.data, <int*>rstat.data)
        return cstat, rstat


def getModelExample():
    '''
    Return a model example to be used in doctests.
    '''
    import numpy as np
    from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
    from cylp.cy import CyClpSimplex

    model = CyLPModel()
    x = model.addVariable('x', 3)
    y = model.addVariable('y', 2)

    A = np.matrix([[1., 2., 0], [1., 0, 1.]])
    B = np.matrix([[1., 0, 0], [0, 0, 1.]])
    D = np.matrix([[1., 2.], [0, 1]])
    a = CyLPArray([5, 2.5])
    b = CyLPArray([4.2, 3])
    x_u= CyLPArray([2., 3.5])

    model += A * x <= a
    model += 2 <= B * x + D * y <= b
    model += y >= 0
    model += 1.1 <= x[1:3] <= x_u

    c = CyLPArray([1., -2., 3.])
    model.objective = c * x + 2 * y.sum()

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
