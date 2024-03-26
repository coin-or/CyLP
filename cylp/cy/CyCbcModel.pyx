# cython: embedsignature=True

from itertools import product
from cylp.py.mip import NodeCompareBase
from cylp.py.modeling.CyLPModel import CyLPSolution
from cylp.cy.CyCutGeneratorPythonBase cimport CyCutGeneratorPythonBase
from libcpp cimport bool


cdef int RunTest(void* ptr, CppICbcNode*x, CppICbcNode*y):
    obj = <object> ptr
    return obj.compare(CyCbcNode().setCppSelf(x),
                       CyCbcNode().setCppSelf(y))

cdef bool RunNewSolution(void* ptr, CppICbcModel* model,
        double objectiveAtContinuous,
        int numberInfeasibilitiesAtContinuous):
    obj = <object> ptr
    return obj.newSolution(CyCbcModel().setCppSelf(model),
                                objectiveAtContinuous,
                                numberInfeasibilitiesAtContinuous)

cdef int RunEvery1000Nodes(void* ptr, CppICbcModel* model, int numberNodes):
    obj = <object> ptr
    return obj.every1000Nodes(CyCbcModel().setCppSelf(model),
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
    >>> s = CyClpSimplex(model)
    >>>
    >>> cbcModel = s.getCbcModel()
    >>>
    >>> cbcModel.solve()
    0
    >>> print (cbcModel.status)
    'solution'
    >>> sol_x = cbcModel.primalVariableSolution['x']
    >>>
    >>> (abs(sol_x - np.array([0, 2, 2])) <= 10**-6).all()
    True

    '''

    def __cinit__(self, cyLPModel=None):
        self.cyLPModel = cyLPModel
        self.cutGenerators = []

    def __dealloc__(self):
        for generator in self.cutGenerators:
            Py_DECREF(generator)

        try:
            if self.CppSelf:
                del self.CppSelf
        except AttributeError:
            pass

    cdef setCppSelf(self, CppICbcModel* cppmodel):
        self.CppSelf = cppmodel
        return self

    cdef setClpModel(self, clpmodel):
        self.clpModel = clpmodel
        return self

    def setNodeCompare(self, nodeCompareObject):
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
        cdef CyCutGeneratorPythonBase generator = \
                            CyCutGeneratorPythonBase(pythonCutGeneratorObject)
        generator.cyLPModel = self.cyLPModel
        self.CppSelf.addCutGenerator(<CppCglCutGenerator*>generator.CppSelf,
                                    howOften, name, normal, atSolution,
                                    infeasible, howOftenInSub, whatDepth,
                                    whatDepthInSub)

    def solve(self):
        '''
        Call CbcMain. Solve the problem using the same parameters used by CbcSolver.
        Equivalent to solving the model from the command line using cbc's binary.
        '''
        return self.CppSelf.cbcMain()

    property status:
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
        def __get__(self):
            return self.CppSelf.logLevel()

        def __set__(self, value):
            self.CppSelf.setLogLevel(value)

    def isRelaxationInfeasible(self):
        return self.CppSelf.isInitialSolveProvenPrimalInfeasible()

    def isRelaxationDualInfeasible(self):
        return self.CppSelf.isInitialSolveProvenDualInfeasible()

    def isRelaxationOptimal(self):
        return self.CppSelf.isInitialSolveProvenOptimal()

    def isRelaxationAbandoned(self):
        return self.CppSelf.isInitialSolveAbandoned()

    property osiSolverInteface:
        def __get__(self):
            cdef CyOsiSolverInterface osi = CyOsiSolverInterface()
            osi.setCppSelf(self.CppSelf.solver())
            return osi

    property primalVariableSolution:
        def __get__(self):
            ret = <object>self.CppSelf.getPrimalVariableSolution()
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
                names = self.clpModel.variableNames
                if names:
                    d = CyLPSolution()
                    for i in range(len(names)):
                        d[names[i]] = ret[i]
                    ret = d
            return ret

    property solutionCount:
        def __get__(self):
            return self.CppSelf.getSolutionCount()

    property numberHeuristicSolutions:
        def __get__(self):
            return self.CppSelf.getNumberHeuristicSolutions()

    property nodeCount:
        def __get__(self):
            return self.CppSelf.getNodeCount()

    property objectiveValue:
        def __get__(self):
            return self.CppSelf.getObjValue()

    property bestPossibleObjValue:
        def __get__(self):
            return self.CppSelf.getBestPossibleObjValue()

    property numberObjects:
        def __get__(self):
            return self.CppSelf.numberObjects()

    property integerTolerance:
        def __get__(self):
            return self.CppSelf.getIntegerTolerance()

        def __set__(self, value):
           self.CppSelf.setIntegerTolerance(value)

    property maximumSeconds:
        def __get__(self):
            return self.CppSelf.getMaximumSeconds()

        def __set__(self, value):
           self.CppSelf.setMaximumSeconds(value)

    property maximumNodes:
        def __get__(self):
            return self.CppSelf.getMaximumNodes()

        def __set__(self, value):
           self.CppSelf.setMaximumNodes(value)

    property numberThreads:
        def __get__(self):
            return self.CppSelf.getNumberThreads()

        def __set__(self, value):
            self.CppSelf.setNumberThreads(value)

    property allowableGap:
        def __get__(self):
            return self.CppSelf.getAllowableGap()

        def __set__(self, value):
            self.CppSelf.setAllowableGap(value)

    property allowableFractionGap:
        def __get__(self):
            return self.CppSelf.getAllowableFractionGap()

        def __set__(self, value):
            self.CppSelf.setAllowableFractionGap(value)

    property allowablePercentageGap:
        def __get__(self):
            return self.CppSelf.getAllowablePercentageGap()

        def __set__(self, value):
            self.CppSelf.setAllowablePercentageGap(value)

    property maximumSolutions:
        def __get__(self):
            return self.CppSelf.getMaximumSolutions()

        def __set__(self, value):
            self.CppSelf.setMaximumSolutions(value)



    #TODO: add access to solver: getLower, getUpper,...
