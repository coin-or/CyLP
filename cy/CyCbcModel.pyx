# cython: embedsignature=True

from itertools import izip, product
from CyLP.py.mip import NodeCompareBase
from CyLP.py.modeling.CyLPModel import CyLPSolution
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
problemStatus =  ['solution', 'relaxation infeasible', 
         'stopped on gap', 'stopped on nodes', 'stopped on time'
         'stopped on user event', 'stopped on solutions'
         'linear relaxation unbounded', 'unset']

cdef class CyCbcModel:
    '''
    Interfaces ``CbcModel``. To solve a first you create a 
    :class:`CyLP.cy.CyClpSimplex` object either
    by reading it from an ``mps`` file using 
    :func:`CyClpSimplex.readMps() <CyLP.cy.CyClpSimplex.CyClpSimplex.readMps>` 
    or by using CyLP modeling tool 
    :mod:`CyLP.py.modeling.CyLPModel`. Then you ask the object for a 
    ``CyCbcModel`` which is capable solving MIPs using B&B

    **Usage**

    >>> import numpy as np
    >>> from CyLP.cy import CyCbcModel, CyClpSimplex
    >>> from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray
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
    >>> cbcModel.branchAndBound()
    'solution'
    >>> sol_x = cbcModel.primalVariableSolution['x']
    >>>
    >>> (abs(sol_x - np.array([0, 2, 2])) <= 10**-6).all()
    True

    '''
    
    def __cinit__(self, cyLPModel=None):
        self.cyLPModel = cyLPModel

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
        self.CppSelf.addCutGenerator(generator.CppSelf, howOften,
                                    name, normal, atSolution,
                                    infeasible, howOftenInSub, whatDepth,
                                    whatDepthInSub)

    def branchAndBound(self, doStatistics=0):
        self.CppSelf.branchAndBound(doStatistics)
        return self.status

    property status:
        def __get__(self):
            if self.isRelaxationInfeasible():
                return problemStatus[1]
            if self.isRelaxationAbondoned():
                return 'relaxation abondoned'

            return problemStatus[self.CppSelf.status()]

    def isRelaxationInfeasible(self):
        return self.CppSelf.isInitialSolveProvenPrimalInfeasible()

    def isRelaxationDualInfeasible(self): 
        return self.CppSelf.isInitialSolveProvenDualInfeasible()
    
    def isRelaxationOptimal(self):
        return self.CppSelf.isInitialSolveProvenOptimal()
    
    def isRelaxationAbondoned(self):
        return self.CppSelf.isInitialSolveAbandoned()

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
        
    #TODO: add access to solver: getLower, getUpper,...
