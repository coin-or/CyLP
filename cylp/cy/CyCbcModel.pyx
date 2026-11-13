# cython: embedsignature=True

from itertools import product
import numpy as np
from cylp.py.mip import NodeCompareBase
from cylp.py.modeling.CyLPModel import CyLPSolution
from cylp.cy.CyCutGeneratorPythonBase cimport CyCutGeneratorPythonBase
from libcpp cimport bool


cdef int RunTest(void* ptr, CppICbcNode*x, CppICbcNode*y):
    obj = <object> ptr
    return obj.compare(CyCbcNode().setCppSelf(x),
                       CyCbcNode().setCppSelf(y))

# The model passed to these callbacks belongs to Cbc, so the wrapper must
# not delete it
cdef bool RunNewSolution(void* ptr, CppICbcModel* model,
        double objectiveAtContinuous,
        int numberInfeasibilitiesAtContinuous):
    obj = <object> ptr
    return obj.newSolution(CyCbcModel().setCppSelf(model, False),
                                objectiveAtContinuous,
                                numberInfeasibilitiesAtContinuous)

cdef int RunEvery1000Nodes(void* ptr, CppICbcModel* model, int numberNodes):
    obj = <object> ptr
    return obj.every1000Nodes(CyCbcModel().setCppSelf(model, False),
                                 numberNodes)

# Understandable messages to translate what branchAndBound() returns
problemStatus =  ['search completed', 'relaxation infeasible',
         'stopped on gap', 'stopped on nodes', 'stopped on time',
         'stopped on user event', 'stopped on solutions',
         'linear relaxation unbounded', 'unset']

cdef class CyCbcModel:
    '''
    Interfaces ``CbcModel``. To solve a first you create a
    :class:`cylp.cy.CyClpSimplex` object either
    by reading it from an ``mps`` file using
    :func:`CyClpSimplex.readMps() <cylp.cy.CyClpSimplex.CyClpSimplex.readMps>`
    or by using cylp modeling tool
    :mod:`cylp.py.modeling.CyLPModel`. Then you ask the object for a
    ``CyCbcModel`` which is capable solving MIPs using B&B

    **Usage**

    >>> import numpy as np
    >>> from cylp.cy import CyCbcModel, CyClpSimplex
    >>> from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
    >>> model = CyLPModel()
    >>>
    >>> x = model.addVariable('x', 3, isInt=True)
    >>> y = model.addVariable('y', 2)
    >>>
    >>> A = np.array([[1., 2., 0],[1., 0, 1.]])
    >>> B = np.array([[1., 0, 0], [0, 0, 1.]])
    >>> D = np.array([[1., 2.],[0, 1]])
    >>> a = CyLPArray([5, 2.5])
    >>> b = CyLPArray([4.2, 3])
    >>> x_u= CyLPArray([2., 3.5])
    >>>
    >>> model += A @ x <= a
    >>> model += 2 <= B @ x + D @ y <= b
    >>> model += y >= 0
    >>> model += 1.1 <= x[1:3] <= x_u
    >>>
    >>> c = CyLPArray([1., -2., 3.])
    >>> model.objective = c * x + 2 * y.sum()
    >>>
    >>> s = CyClpSimplex(model)
    >>>
    >>> cbcModel = s.getCbcModel()
    >>>
    >>> cbcModel.solve()
    0
    >>> print(cbcModel.status)
    solution
    >>> sol_x = cbcModel.primalVariableSolution['x']
    >>>
    >>> (abs(sol_x - np.array([0, 2, 2])) <= 10**-6).all()
    True

    '''

    def __cinit__(self, cyLPModel=None):
        self.cyLPModel = cyLPModel
        self.cutGenerators = []
        self.pythonCutGenerators = []

    def __dealloc__(self):
        for generator in self.cutGenerators:
            Py_DECREF(generator)

        try:
            if self.CppSelf and self.ownCppSelf:
                del self.CppSelf
        except AttributeError:
            pass

    cdef setCppSelf(self, CppICbcModel* cppmodel, bint own=True):
        self.CppSelf = cppmodel
        self.ownCppSelf = own
        return self

    cdef setClpModel(self, clpmodel):
        self.clpModel = clpmodel
        return self

    def setNodeCompare(self, nodeCompareObject):
        '''
        Use ``nodeCompareObject``, a :class:`cylp.py.mip.NodeCompareBase`, to
        choose the next node of the branch-and-bound tree to explore.
        '''
        if not isinstance(nodeCompareObject, NodeCompareBase):
            raise TypeError('setNodeCompare argument should be a ' \
                        'NodeCompareBase object. Got %s' %
                        nodeCompareObject.__class__)

        self.CppSelf.setNodeCompare(<PyObject*>nodeCompareObject,
                                    RunTest, RunNewSolution, RunEvery1000Nodes)

    cpdef addCutGenerator(self, CyCglCutGenerator generator,
                        howOften=1, name="", normal=True, atSolution=False,
                        infeasible=False, howOftenInSub=-100, whatDepth=-1,
                        whatDepthInSub=-1):
        '''
        Add a Cgl cut generator, such as :class:`cylp.cy.CyCgl.CyCglGomory`. The
        other arguments are those of ``CbcModel::addCutGenerator``.
        '''
        self.cutGenerators.append(generator)
        Py_INCREF(generator)
        if isinstance(name, str):
            # Cast strings/unicode to bytes
            name = name.encode('utf-8')
        self.CppSelf.addCutGenerator(generator.CppSelf, howOften,
                                    name, normal, atSolution,
                                    infeasible, howOftenInSub, whatDepth,
                                    whatDepthInSub)

    def addPythonCutGenerator(self, pythonCutGeneratorObject,
                        howOften=1, name="", normal=True, atSolution=False,
                        infeasible=False, howOftenInSub=-100, whatDepth=-1,
                        whatDepthInSub=-1):
        '''
        Add a cut generator written in Python: an object whose
        ``generateCuts(si, cglTreeInfo)`` method returns a list of cuts. See
        ``cylp/py/mip/GomoryCutGenerator.py`` for an example. The other
        arguments are those of :py:meth:`addCutGenerator`.

        The cuts must be in terms of the model's variables, but Cbc calls
        ``generateCuts`` on the problem after its preprocessing, which can
        have fewer rows and columns. See
        `issue 255 <https://github.com/coin-or/CyLP/issues/255>`__.

        If ``generateCuts`` raises an exception, Cbc finishes its search
        without calling it again, and :py:meth:`solve` then raises the
        exception.
        '''
        cdef CyCutGeneratorPythonBase generator = \
                            CyCutGeneratorPythonBase(pythonCutGeneratorObject)
        generator.cyLPModel = self.cyLPModel
        self.pythonCutGenerators.append(generator)
        if isinstance(name, str):
            name = name.encode('utf-8')
        self.CppSelf.addCutGenerator(<CppCglCutGenerator*>generator.CppSelf,
                                    howOften, name, normal, atSolution,
                                    infeasible, howOftenInSub, whatDepth,
                                    whatDepthInSub)

    def solve(self):
        '''
        Call CbcMain. Solve the problem using the same parameters used by CbcSolver.
        Equivalent to solving the model from the command line using cbc's binary.
        '''
        cdef CyCutGeneratorPythonBase generator
        for generator in self.pythonCutGenerators:
            generator.exception = None
        ret = self.CppSelf.cbcMain()
        for generator in self.pythonCutGenerators:
            if generator.exception is not None:
                e, generator.exception = generator.exception, None
                raise e
        return ret

    property status:
        '''
        The result of :py:meth:`solve`:

        - ``'solution'`` if Cbc proved that its solution is optimal, or within
          :py:attr:`allowableGap` or :py:attr:`allowableFractionGap` of the
          optimum,
        - ``'relaxation infeasible'``, ``'relaxation abandoned'`` or
          ``'problem proven infeasible'``,
        - otherwise, why Cbc stopped: ``'stopped on gap'``,
          ``'stopped on nodes'``, ``'stopped on time'``,
          ``'stopped on user event'``, ``'stopped on solutions'``,
          ``'linear relaxation unbounded'``, ``'search completed'`` or
          ``'unset'``.
        '''
        def __get__(self):
            # secondaryStatus() should be used instead of status() (??)
            if self.isRelaxationInfeasible():
               return problemStatus[1]
            if self.isRelaxationAbandoned():
               return 'relaxation abandoned'
            if self.CppSelf.isProvenInfeasible():
               return 'problem proven infeasible'
            if self.CppSelf.isProvenOptimal():
               return 'solution'
            return problemStatus[self.CppSelf.secondaryStatus()]

    property logLevel:
        '''
        How much Cbc prints: 0 for nothing, more for more.
        '''
        def __get__(self):
            return self.CppSelf.logLevel()

        def __set__(self, value):
            self.CppSelf.setLogLevel(value)

    def isRelaxationInfeasible(self):
        '''
        Whether Cbc proved that the LP relaxation it solved first is infeasible.
        '''
        return self.CppSelf.isInitialSolveProvenPrimalInfeasible()

    def isRelaxationDualInfeasible(self):
        '''
        Whether Cbc proved that the LP relaxation it solved first is dual
        infeasible, i.e. unbounded if it's feasible.
        '''
        return self.CppSelf.isInitialSolveProvenDualInfeasible()

    def isRelaxationOptimal(self):
        '''
        Whether Cbc solved the LP relaxation it solved first to optimality.
        '''
        return self.CppSelf.isInitialSolveProvenOptimal()

    def isRelaxationAbandoned(self):
        '''
        Whether Cbc gave up on the LP relaxation it solved first because of
        numerical difficulties.
        '''
        return self.CppSelf.isInitialSolveAbandoned()

    property osiSolverInteface:
        '''
        The solver that Cbc uses for the LP relaxations, as a
        :class:`cylp.cy.CyOsiSolverInterface.CyOsiSolverInterface`.
        '''
        def __get__(self):
            cdef CyOsiSolverInterface osi = CyOsiSolverInterface()
            osi.setCppSelf(self.CppSelf.solver())
            return osi

    property primalVariableSolution:
        '''
        Solution to the primal variables.

        :rtype: Numpy array
        '''
        def __get__(self):
            return self._byVariable(
                    <object>self.CppSelf.getPrimalVariableSolution())

    property primalColumnSolution:
        '''
        Solution to the primal variables as one array, in column order.
        Unlike :py:attr:`primalVariableSolution`, it is not split up by
        variable, which is much faster for large models. A copy.

        :rtype: Numpy array
        '''
        def __get__(self):
            return np.array(<object>self.CppSelf.getPrimalVariableSolution())

    property dualVariableSolution:
        '''
        Variables' reduced costs. See :py:attr:`dualConstraintSolution` for
        the LP that they belong to.

        :rtype: Numpy array
        '''
        def __get__(self):
            return self._byVariable(self.dualColumnSolution)

    property dualColumnSolution:
        '''
        Variables' reduced costs as one array, in column order. Unlike
        :py:attr:`dualVariableSolution`, it is not split up by variable.
        A copy.

        :rtype: Numpy array
        '''
        def __get__(self):
            return np.array(<object>self.CppSelf.getDualColumnSolution())

    property primalConstraintSolution:
        '''
        Constraints' activities, i.e. the value of each constraint's
        left-hand side at the solution.

        :rtype: Numpy array
        '''
        def __get__(self):
            return self._byConstraint(
                    np.array(<object>self.CppSelf.getPrimalRowSolution()))

    property dualConstraintSolution:
        '''
        Dual variables' solution. Once :py:meth:`solve` has found a
        solution, Cbc fixes the integer variables at their values in it and
        solves the LP relaxation again, so these are the duals of that LP.

        :rtype: Numpy array
        '''
        def __get__(self):
            return self._byConstraint(
                    np.array(<object>self.CppSelf.getDualRowSolution()))

    def _byVariable(self, ret):
        '''
        Split the column array ``ret`` up by the CyLPModel's variables, or
        by column name if there is no model. If the columns have no names
        either, return ``ret`` itself.
        '''
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
                        d[v][element] = ret[var.fromInd +
                                            var[element].indices[0]]
            return d
        names = self.clpModel.variableNames
        if names:
            d = CyLPSolution()
            for i in range(len(names)):
                d[names[i]] = ret[i]
            return d
        return ret

    def _byConstraint(self, ret):
        '''
        Split the row array ``ret`` up by constraint.
        '''
        if self.cyLPModel:
            inds = self.cyLPModel.inds
            return {c: ret[inds.constIndex[c]] for c in inds.constIndex}
        return ret

    property solutionCount:
        '''
        The number of solutions that Cbc found.
        '''
        def __get__(self):
            return self.CppSelf.getSolutionCount()

    property numberHeuristicSolutions:
        '''
        The number of solutions that Cbc's heuristics found.
        '''
        def __get__(self):
            return self.CppSelf.getNumberHeuristicSolutions()

    property nodeCount:
        '''
        The number of nodes of the branch-and-bound tree that Cbc explored.
        '''
        def __get__(self):
            return self.CppSelf.getNodeCount()

    property objectiveValue:
        '''
        The objective value of the best solution found.
        '''
        def __get__(self):
            return self.CppSelf.getObjValue()

    property bestPossibleObjValue:
        '''
        A bound on the optimal objective value: the best of the best possible
        value left in the branch-and-bound tree and the best solution found.
        '''
        def __get__(self):
            return self.CppSelf.getBestPossibleObjValue()

    property numberObjects:
        '''
        The number of objects that Cbc branches on, such as integer variables.
        It is 0 until :py:meth:`solve` creates them.
        '''
        def __get__(self):
            return self.CppSelf.numberObjects()

    property integerTolerance:
        '''
        How far from an integer an integer variable's value can be and still
        count as an integer.
        '''
        def __get__(self):
            return self.CppSelf.getIntegerTolerance()

        def __set__(self, value):
           self.CppSelf.setIntegerTolerance(value)

    property maximumSeconds:
        '''
        The time limit of :py:meth:`solve`, in seconds.
        '''
        def __get__(self):
            return self.CppSelf.getMaximumSeconds()

        def __set__(self, value):
           self.CppSelf.setMaximumSeconds(value)

    property maximumNodes:
        '''
        The maximum number of nodes that :py:meth:`solve` explores.
        '''
        def __get__(self):
            return self.CppSelf.getMaximumNodes()

        def __set__(self, value):
           self.CppSelf.setMaximumNodes(value)

    property numberThreads:
        '''
        The number of threads that Cbc uses, if it was built with threads.
        0 means the default.
        '''
        def __get__(self):
            return self.CppSelf.getNumberThreads()

        def __set__(self, value):
            self.CppSelf.setNumberThreads(value)

    property allowableGap:
        '''
        Stop when the gap between the objective value of the best solution
        and :py:attr:`bestPossibleObjValue` is less than this.
        '''
        def __get__(self):
            return self.CppSelf.getAllowableGap()

        def __set__(self, value):
            self.CppSelf.setAllowableGap(value)

    property allowableFractionGap:
        '''
        Stop when the gap between the objective value of the best solution
        and :py:attr:`bestPossibleObjValue` is less than this fraction of the
        absolute value of the best solution's objective value.
        '''
        def __get__(self):
            return self.CppSelf.getAllowableFractionGap()

        def __set__(self, value):
            self.CppSelf.setAllowableFractionGap(value)

    property allowablePercentageGap:
        '''
        :py:attr:`allowableFractionGap` as a percentage.
        '''
        def __get__(self):
            return self.CppSelf.getAllowablePercentageGap()

        def __set__(self, value):
            self.CppSelf.setAllowablePercentageGap(value)

    property maximumSolutions:
        '''
        Stop after finding this many solutions.
        '''
        def __get__(self):
            return self.CppSelf.getMaximumSolutions()

        def __set__(self, value):
            self.CppSelf.setMaximumSolutions(value)



    #TODO: add access to solver: getLower, getUpper,...
