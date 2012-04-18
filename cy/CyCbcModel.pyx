# cython: embedsignature=True

from CyLP.py.mip import NodeCompareBase
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
    >>> model.addConstraint(A*x <= a)
    >>> model.addConstraint(2 <= B * x + D * y <= b)
    >>> model.addConstraint(y >= 0)
    >>> model.addConstraint(1.1 <= x[1:3] <= x_u)
    >>>
    >>> c = CyLPArray([1., -2., 3.])
    >>> model.objective = c * x + 2 * y
    >>>
    >>> s = CyClpSimplex(model)
    >>>
    >>> cbcModel = s.getCbcModel()
    >>>
    >>> cbcModel.branchAndBound()
    >>>
    >>> sol = cbcModel.primalVariableSolution
    >>>
    >>> (abs(sol - np.array([0, 2, 2, 0, 1])) <= 10**-6).all()
    True

    '''
    
    cdef setCppSelf(self, CppICbcModel* cppmodel):
        self.CppSelf = cppmodel
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

    property primalVariableSolution:
        def __get__(self):
            return <object>self.CppSelf.getPrimalVariableSolution()

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

    #TODO: add access to solver: getLower, getUpper,...
