import sys
import cProfile
import inspect
from time import clock
import numpy as np
from scipy import sparse
from CyLP.cy import CyClpSimplex
from QPSReader import readQPS
from CyLP.py.pivots.WolfePivot import WolfePivot
from CyLP.py.pivots.PositiveEdgeWolfePivot import PositiveEdgeWolfePivot
from CyCoinModel import CyCoinModel
from CyLP.py.utils.sparseUtil import csc_matrixPlus


def getSolution(s, varGroupname):
    sol = s.getPrimalVariableSolution()
    return np.array([sol[i] for i in IndexFactory.varIndex[varGroupname]])


## A generic QP class
class QP:

    def __init__(self):
        # div by 2 just to avoid numeric errors
        self.infinity = sys.float_info.max / 2

    def ComputeObjectiveValue(self, x):
        return (0.5 * x * (x * self.G).T + np.dot(self.c, x) -
                self.objectiveOffset)

    def init(self, G, c, A, b, C, c_low, c_up, x_low, x_up):
        self.c = c.copy()
        self.A = A.copy()
        self.b = b.copy()
        self.n = G.shape[0]
        self.m = b.shape[0] + C.shape[0]
        self.nInEquality = C.shape[0]
        self.nEquality = b.shape[0]
        self.G = G.copy()
        self.C = C.copy()
        self.c_low = c_low.copy()
        self.c_up = c_up.copy()
        self.x_low = x_low.copy()
        self.x_up = x_up.copy()

    def sAll(self, x):
        '''
        Return Cx - b
        '''
        return self.A * x - self.b

    def s(self, x, i):
        '''
        Return C_ix - b_i
        :arg x: a vector
        :type x: Numpy array
        :arg i: index
        :type x: integer
        '''
        if i >= self.m:
            raise "i should be smaller than m"
        return self.A[i, :] * x - self.b[i]

    def getUnconstrainedSol(self):
        '''
        Return the unconstrained minimizer of the QP, :math:`-G^{-1}a`
        '''
        #G = self.L * self.L.T
        #Change the following line and use self.L
        return np.linalg.solve(G, -self.c)

    def gradient(self, x):
        '''
        Return the gradient of the objective function at ``x``
        
        '''
        return self.G * x + self.a

    def fromQps(self, filename):
        (self.G, self.c, self.A, self.b,
            self.C, self.c_low, self.c_up,
            self.x_low, self.x_up,
            self.n, self.nEquality, self.nInEquality,
            self.objectiveOffset) = readQPS(filename)

    def Wolfe_2(self):
        "Solves a QP using Wolfe's method"
        A = self.A
        G = self.G
        b = self.b
        c = self.c
        C = self.C
        c_low = self.c_low
        c_up = self. c_up
        x_low = self.x_low
        x_up = self.x_up
        infinity = self.infinity

        nVar = self.n
        nEquality = self.nEquality
        nInEquality = self.nInEquality
        nCons = nEquality + nInEquality

        varIndexDic = {}
        currentNumberOfVars = 0
        constIndexDic = {}
        currentNumberOfConst = 0

        s = CyClpSimplex()
        model = CyCoinModel()

        inds = xrange(nVar)

        # Convert G and A to sparse matrices (coo) if they're not already
#       if type(G) == np.matrixlib.defmatrix.matrix:
#           temp = sparse.lil_matrix(G.shape)
#           temp[:, :] = G
#           G = temp.tocoo()
#
#       if nEquality != 0 and type(A) == np.matrixlib.defmatrix.matrix:
#           temp = sparse.lil_matrix(A.shape)
#           temp[:, :] = A
#           A = temp.tocoo()
#
#       if nInEquality != 0 and type(C) == np.matrixlib.defmatrix.matrix:
#           temp = sparse.lil_matrix(C.shape)
#           temp[:, :] = C
#           C = temp.tocoo()

        #i for indices, n for size of the set
        iVarsWithUpperBound = \
                [i for i in xrange(len(x_up)) if x_up[i] < infinity]
        nVarsWithUpperBound = len(iVarsWithUpperBound)
        iVarsWithLowerBound = \
                [i for i in xrange(len(x_low)) if x_low[i] > -infinity]
        nVarsWithLowerBound = len(iVarsWithLowerBound)

        iConstraintsWithUpperLowerBound = \
                [i for i in xrange(nInEquality)
                        if c_up[i] < infinity and c_low[i] > -infinity]
        nConstraintsWithUpperLowerBound = len(iConstraintsWithUpperLowerBound)

        iConstraintsWithJustUpperBound = \
                [i for i in xrange(nInEquality)
                        if c_up[i] < infinity and c_low[i] <= -infinity]
        nConstraintsWithJustUpperBound = len(iConstraintsWithJustUpperBound)

        iConstraintsWithJustLowerBound = \
                [i for i in xrange(nInEquality)
                        if c_up[i] >= infinity and c_low[i] > -infinity]
        nConstraintsWithJustLowerBound = len(iConstraintsWithJustLowerBound)

        iVarsWithUpperBound = [i for i in inds if x_up[i] < infinity]
        nVarsWithUpperBound = len(iVarsWithUpperBound)

        iVarsWithLowerBound = [i for i in inds if x_low[i] > -infinity]
        nVarsWithLowerBound = len(iVarsWithLowerBound)

        # AA for Augmented A
        # number of elements of x, g1, g2, g3, x_u, x_l, g_u, s^+, s^-
        nColsAA = nVar + \
                    nInEquality + \
                    nVarsWithLowerBound + \
                    nVarsWithUpperBound + \
                    nConstraintsWithUpperLowerBound + \
                    nVar + \
                    nVar

        s.resize(0, nColsAA)

        #adding Ax = b
        zeroArray = np.zeros(nVar)
        infinityArray = infinity + zeroArray
        #minusInfinityArray = -infinity + zeroArray

        if nEquality:
            #s.addColumns(A.getnnz(), zeroArray,
            #        infinityArray, zeroArray, A.indptr, A.indices, A.data)
            s.addRows(nEquality, b, b, A.indptr, A.indices, A.data)

        if nInEquality:
            CRhs = np.zeros(nInEquality)
            nCol_C = C.shape[1]
            for i in range(nInEquality):
                if i in iConstraintsWithUpperLowerBound:
                    C[i, nCol_C + i] = 1
                    CRhs[i] = 1
                elif i in iConstraintsWithJustUpperBound:
                    C[i, nCol_C + i] = 1
                    CRhs[i] = 1
                else:
                    C[i, nCol_C + i] = -1
                    CRhs[i] = -1

            s.addRows(nInEquality, CRhs, CRhs, C.indptr, C.indices, C.data)

        return

        #constructing A for Ax = b
        # step one: concatenating C to the bottom

        #but first adding the indices of Ax=b to the Index Factory
        IndexFactory.addConst('Ax=b', nEquality)
        IndexFactory.addVar('x', nVar)

        for i in range(nInEquality):
            rowi = C.getrow(i).tocoo()
            if c_up[i] < infinity:
                rowi = sparseConcat(rowi, e_sparse(i, nInEquality, 1), 'h')
                A = sparseConcat(A, rowi, 'v')
                #if i in c_low.keys():
                if c_low[i] > -infinity:
                    b = np.concatenate((b, np.array([c_up[i]])), axis=1)
                    IndexFactory.addConst('C_1X+g1=c_up', 1)
                    IndexFactory.addVar('g1', 1)
                else:
                    b = np.concatenate((b, np.array([c_up[i]])), axis=1)
                    IndexFactory.addConst('C_2X+g2=c_up', 1)
                    IndexFactory.addVar('g2', 1)
            else:
                rowi = sparseConcat(rowi, e_sparse(i, nInEquality, -1), 'h')
                A = sparseConcat(A, rowi, 'v')
                b = np.concatenate((b, np.array([c_low[i]])), axis=1)
                IndexFactory.addConst('C_3X-g3=c_low', 1)
                IndexFactory.addVar('g3', 1)

            #whatever the variable (g1,g2 or g3) >=0

        ##step two: adding x + x_u <= U and x - x_l >= L
        x_up_count = 0

        lengthOfRowToAdd = IndexFactory.currentVarIndex + \
                                        len(iVarsWithUpperBounds)
        startInd = IndexFactory.currentVarIndex
        for i in iVarsWithUpperBounds:
                rowToAdd = e_sparse(i, lengthOfRowToAdd, 1)
                rowToAdd[0, startInd + x_up_count] = 1
                x_up_count += 1
                A = sparseConcat(A, rowToAdd, 'v')
                b = np.concatenate((b, np.array([x_up[i]])), axis=1)
                IndexFactory.addConst('x+x_u=x_up', 1)
                IndexFactory.addVar('x_u', 1)

        lengthOfRowToAdd = IndexFactory.currentVarIndex + \
                                            len(iVarsWithLowerBounds)

        startInd = IndexFactory.currentVarIndex
        x_low_count = 0
        #for i in x_low.keys():
        for i in iVarsWithLowerBounds:
                rowToAdd = e_sparse(i, lengthOfRowToAdd, 1)
                rowToAdd[0,  startInd + x_low_count] = -1
                x_low_count += 1
                A = sparseConcat(A, rowToAdd, 'v')
                b = np.concatenate((b, np.array([x_low[i]])), axis=1)
                IndexFactory.addConst('x-x_l=x_low', 1)
                IndexFactory.addVar('x_l', 1)

        if 'g1' in IndexFactory.varIndex.keys():
            g1_inds = IndexFactory.varIndex['g1']
            for i in range(len(g1_inds)):
                g1_i = g1_inds[i]
                rowToAdd = e_sparse(g1_i,
                                IndexFactory.currentVarIndex + len(g1_inds), 1)
                rowToAdd[0, IndexFactory.currentVarIndex + i] = 1
                A = sparseConcat(A, rowToAdd, 'v')
                indexInC = IndexFactory.constIndex['C_1X+g1=c_up'][i] - \
                                                            nEquality
                b = np.concatenate((b,
                        np.array([c_up[indexInC] - c_low[indexInC]])), axis=1)
                IndexFactory.addConst('g_1+g_u=c_up-c_low', 1)
                IndexFactory.addVar('g_u', 1)

        #(Augmented A) x = b
        for i in range(A.shape[0]):
            rowi = A.getrow(i).tocoo()
            model.addConstraint(rowi.nnz,
                    np.array(rowi.col, np.int32),
                    np.array(rowi.data, 'd'), b[i], b[i])

        setPositive(model, 'g1')
        setPositive(model, 'g2')
        setPositive(model, 'g3')
        setPositive(model, 'x_u')
        setPositive(model, 'x_l')
        setPositive(model, 'g_u')

        #Gx = -c
        for i in range(nVar):
            rowi = G.getrow(i).tocoo()
            model.addConstraint(rowi.nnz,
                    np.array(rowi.col, np.int32),
                    np.array(rowi.data, np.double), -c[i], -c[i])
        IndexFactory.addConst('Gx=-c', nVar)

        for i in range(nVar):
            model.setVariableLower(i, -infinity)

        #adding s^+    so we have: GX + s^+ = -c
        for i in range(nVar):
            model.addVariable(1,
                        np.array([IndexFactory.constIndex['Gx=-c'][i]],
                        np.int32), np.array([1.], np.double),
                        0, infinity, 0)
        IndexFactory.addVar('s^+', nVar)

        #adding s^-    now: GX + s^+ - s^- = -c
        for i in range(nVar):
            model.addVariable(1,
                        np.array([IndexFactory.constIndex['Gx=-c'][i]],
                        np.int32),
                        np.array([-1.], np.double), 0, infinity, 0)
        IndexFactory.addVar('s^-', nVar)

        # We can run the primal here to find a feasible
        # point to start the second phase
        # as in Wolfe. But in practice, doesn't seem like a good idea
        #s.loadProblem(model, 0)
        #s.primal()

        #adding extra 0's to G for variables other than x: g1, g2, x_u,...

        # adding rhs=0 for A^Ty constraints corresponding
        # to all variables except 'x'
        # These are the variables for which G.row(i) and c_i are zero
        # we don't know how many they are as, for example, we may or may not
        # have Cx<=c_up or Cx >= c_low or ...
        for i in range(A.shape[1] - nVar):
            model.addConstraint(0, np.array([], np.int32),
                                np.array([], np.double), 0, 0)

        #adding -A^Ty_A
        for i in range(A.shape[0]):
            rowi = -A.getrow(i).tocoo()
            #print 'rowi : '
            #print rowi.col
            cols = rowi.col + IndexFactory.constIndex['Gx=-c'][0]
            #print cols
            model.addVariable(rowi.nnz, np.array(cols, np.int32),
                            np.array(rowi.data, 'd'), -infinity, infinity, 0)
        IndexFactory.addVar('y_A', A.shape[0])

        cl = np.array(range(IndexFactory.currentVarIndex +
                            A.shape[1]), np.int32)
        s.setComplementarityList(cl)

        #adding -z
        startRow = IndexFactory.constIndex['Gx=-c'][0]
        #TODO: check this range
        #for i in range(nVar, A.shape[1]):
        for i in range(nVar, A.shape[1]):
            model.addVariable(1, np.array([startRow + i], np.int32),
                            np.array([-1.], 'd'), 0, infinity, 0)
            IndexFactory.addVar('z', 1)
            compind = IndexFactory.getLastVarIndex()
            s.setComplement(i, compind)
            #cl[i] , cl[compind] = compind, i

        #setting the objective coefficients of s^+ to one,
        sPlusIndex = IndexFactory.varIndex['s^+']
        sMinusIndex = IndexFactory.varIndex['s^-']
        for i in sPlusIndex:
            model.setObjective(i, 1)

        #setting the objective coefficients of s^- to one,
        for i in sMinusIndex:
            model.setObjective(i, 1)

        s.loadProblem(model, 0)

        # I think that preSolve could be harmful here.
        # what happens to the complement variables?
        # s = s.preSolve(feasibilityTolerance = 10**-8)
        # s.setComplementarityList(cl)

        # This means that we want to use IClpSimplexPrimal_Wolfe
        # instead of ClpSimplexPrimal
        s.useCustomPrimal(1)

        #this means that we want to use cythons's Cywolfe
        #s.setPrimalColumnPivotAlgorithmToWolfe()
        #p = WolfePivot(s, bucketSize=float(sys.argv[2]))

        p = PositiveEdgeWolfePivot(s, bucketSize=float(sys.argv[2]))
        s.setPivotMethod(p)

        st = clock()
        s.primal()
        print "CLP time : %g seconds" % (clock() - st)

        x = s.getPrimalVariableSolution()
        print "sol = "
        x = x[:nVar]
        #print x
        G = G.todense()
        #print G.shape
        #print x * G
        #print c
        #print x
        print 0.5 * x * (x * G).T + np.dot(c, x) - self.objectiveOffset

        return

    def Wolfe(self):
        '''
        Solves a QP using Wolfe's method
        '''
        A = self.A
        G = self.G
        b = self.b
        c = self.c
        C = self.C
        c_low = self.c_low
        c_up = self. c_up
        x_low = self.x_low
        x_up = self.x_up
        infinity = self.infinity

        nVar = self.n
        nEquality = self.nEquality
        nInEquality = self.nInEquality

        varIndexDic = {}
        currentNumberOfVars = 0
        constIndexDic = {}
        currentNumberOfConst = 0

        s = CyClpSimplex()
        model = CyCoinModel()

        inds = range(nVar)

        # Convert G and A to sparse matrices (coo) if they're not already
        if type(G) == np.matrixlib.defmatrix.matrix:
            temp = sparse.lil_matrix(G.shape)
            temp[:, :] = G
            G = temp.tocoo()

        if nEquality != 0 and type(A) == np.matrixlib.defmatrix.matrix:
            temp = sparse.lil_matrix(A.shape)
            temp[:, :] = A
            A = temp.tocoo()

        if nInEquality != 0 and type(C) == np.matrixlib.defmatrix.matrix:
            temp = sparse.lil_matrix(C.shape)
            temp[:, :] = C
            C = temp.tocoo()

        #constructing A for Ax = b
        # step one: concatenating C to the bottom

        #but first adding the indices of Ax=b to the Index Factory
        IndexFactory.addConst('Ax=b', nEquality)
        IndexFactory.addVar('x', nVar)

        for i in range(nInEquality):
            rowi = C.getrow(i).tocoo()
            if c_up[i] < infinity:
                rowi = sparseConcat(rowi, e_sparse(i, nInEquality, 1), 'h')
                A = sparseConcat(A, rowi, 'v')
                #if i in c_low.keys():
                if c_low[i] > -infinity:
                    b = np.concatenate((b, np.array([c_up[i]])), axis=1)
                    IndexFactory.addConst('C_1X+g1=c_up', 1)
                    IndexFactory.addVar('g1', 1)
                else:
                    b = np.concatenate((b, np.array([c_up[i]])), axis=1)
                    IndexFactory.addConst('C_2X+g2=c_up', 1)
                    IndexFactory.addVar('g2', 1)
            else:
                rowi = sparseConcat(rowi, e_sparse(i, nInEquality, -1), 'h')
                A = sparseConcat(A, rowi, 'v')
                b = np.concatenate((b, np.array([c_low[i]])), axis=1)
                IndexFactory.addConst('C_3X-g3=c_low', 1)
                IndexFactory.addVar('g3', 1)

            #whatever the variable (g1,g2 or g3) >=0

        ##step two: adding x + x_u <= U and x - x_l >= L
        x_up_count = 0
        iVarsWithUpperBounds = [i for i in range(len(x_up))
                                    if x_up[i] < infinity]

        lengthOfRowToAdd = (IndexFactory.currentVarIndex +
                            len(iVarsWithUpperBounds))
        startInd = IndexFactory.currentVarIndex
        for i in iVarsWithUpperBounds:
                rowToAdd = e_sparse(i, lengthOfRowToAdd, 1)
                rowToAdd[0, startInd + x_up_count] = 1
                x_up_count += 1
                A = sparseConcat(A, rowToAdd, 'v')
                b = np.concatenate((b, np.array([x_up[i]])), axis=1)
                IndexFactory.addConst('x+x_u=x_up', 1)
                IndexFactory.addVar('x_u', 1)

        iVarsWithLowerBounds = [i for i in range(len(x_low))
                                            if x_low[i] > -infinity]

        lengthOfRowToAdd = (IndexFactory.currentVarIndex +
                                len(iVarsWithLowerBounds))

        startInd = IndexFactory.currentVarIndex
        x_low_count = 0
        #for i in x_low.keys():
        for i in iVarsWithLowerBounds:
                rowToAdd = e_sparse(i, lengthOfRowToAdd, 1)
                rowToAdd[0,  startInd + x_low_count] = -1
                x_low_count += 1
                A = sparseConcat(A, rowToAdd, 'v')
                b = np.concatenate((b, np.array([x_low[i]])), axis=1)
                IndexFactory.addConst('x-x_l=x_low', 1)
                IndexFactory.addVar('x_l', 1)

        if 'g1' in IndexFactory.varIndex.keys():
            g1_inds = IndexFactory.varIndex['g1']
            for i in range(len(g1_inds)):
                g1_i = g1_inds[i]
                rowToAdd = e_sparse(g1_i,
                            IndexFactory.currentVarIndex + len(g1_inds), 1)
                rowToAdd[0, IndexFactory.currentVarIndex + i] = 1
                A = sparseConcat(A, rowToAdd, 'v')
                indexInC = IndexFactory.constIndex['C_1X+g1=c_up'][i] - \
                                                                    nEquality
                b = np.concatenate((b, np.array([c_up[indexInC] -
                                    c_low[indexInC]])), axis=1)
                IndexFactory.addConst('g_1+g_u=c_up-c_low', 1)
                IndexFactory.addVar('g_u', 1)

        #(Augmented A) x = b
        for i in range(A.shape[0]):
            rowi = A.getrow(i).tocoo()
            model.addConstraint(rowi.nnz, np.array(rowi.col, np.int32),
                                np.array(rowi.data, 'd'), b[i], b[i])

        setPositive(model, 'g1')
        setPositive(model, 'g2')
        setPositive(model, 'g3')
        setPositive(model, 'x_u')
        setPositive(model, 'x_l')
        setPositive(model, 'g_u')

        #Gx = -c
        for i in range(nVar):
            rowi = G.getrow(i).tocoo()
            model.addConstraint(rowi.nnz, np.array(rowi.col, np.int32),
                        np.array(rowi.data, np.double), -c[i], -c[i])
        IndexFactory.addConst('Gx=-c', nVar)

        for i in range(nVar):
            model.setVariableLower(i, -infinity)

        #adding s^+    so we have: GX + s^+ = -c
        for i in range(nVar):
            model.addVariable(1, np.array(
                        [IndexFactory.constIndex['Gx=-c'][i]], np.int32),
                        np.array([1.], np.double), 0, infinity, 0)
        IndexFactory.addVar('s^+', nVar)

        #adding s^-    now: GX + s^+ - s^- = -c
        for i in range(nVar):
            model.addVariable(1, np.array([
                        IndexFactory.constIndex['Gx=-c'][i]], np.int32),
                        np.array([-1.], np.double), 0, infinity, 0)
        IndexFactory.addVar('s^-', nVar)

        # We can run the primal here to find a feasible point
        # to start the second phase
        # as in Wolfe. But in practice, doesn't seem like a good idea
        #s.loadProblem(model, 0)
        #s.primal()

        #adding extra 0's to G for variables other than x: g1, g2, x_u,...

        #adding rhs=0 for A^Ty constraints corresponding to
        # all variables except 'x'
        #These are the variables for which G.row(i) and c_i are zero
        #we don't know how many they are as, for example, we may or may not
        #have Cx<=c_up or Cx >= c_low or ...
        for i in range(A.shape[1] - nVar):
            model.addConstraint(0, np.array([], np.int32),
                                    np.array([], np.double), 0, 0)

        #adding -A^Ty_A
        for i in range(A.shape[0]):
            rowi = -A.getrow(i).tocoo()
            #print 'rowi : '
            #print rowi.col
            cols = rowi.col + IndexFactory.constIndex['Gx=-c'][0]
            #print cols
            model.addVariable(rowi.nnz, np.array(cols, np.int32),
                            np.array(rowi.data, 'd'), -infinity, infinity, 0)
        IndexFactory.addVar('y_A', A.shape[0])

        cl = np.array(range(IndexFactory.currentVarIndex +
                            A.shape[1]), np.int32)
        s.setComplementarityList(cl)

        #adding -z
        startRow = IndexFactory.constIndex['Gx=-c'][0]
        #TODO: check this range
        #for i in range(nVar, A.shape[1]):
        for i in range(nVar, A.shape[1]):
            model.addVariable(1, np.array([startRow + i], np.int32),
                            np.array([-1.], 'd'), 0, infinity, 0)
            IndexFactory.addVar('z', 1)
            compind = IndexFactory.getLastVarIndex()
            s.setComplement(i, compind)
            #cl[i] , cl[compind] = compind, i

        #setting the objective coefficients of s^+ to one,
        sPlusIndex = IndexFactory.varIndex['s^+']
        sMinusIndex = IndexFactory.varIndex['s^-']
        for i in sPlusIndex:
            model.setObjective(i, 1)

        #setting the objective coefficients of s^- to one,
        for i in sMinusIndex:
            model.setObjective(i, 1)

        s.loadProblem(model, 0)

        # I think that preSolve could be harmful here.
        # what happens to the complement variables?
        #s = s.preSolve(feasibilityTolerance = 10**-8)
        #s.setComplementarityList(cl)

        # This means that we want to use IClpSimplexPrimal_Wolfe
        # instead of ClpSimplexPrimal
        s.useCustomPrimal(1)

        #this means that we want to use cythons's Cywolfe
        #s.setPrimalColumnPivotAlgorithmToWolfe()
        p = WolfePivot(s, bucketSize=float(sys.argv[2]))

        #p = PositiveEdgeWolfePivot(s, bucketSize=float(sys.argv[2]))
        s.setPivotMethod(p)

        st = clock()
        s.primal()
        print "CLP time : %g seconds" % (clock() - st)

        x = s.getPrimalVariableSolution()
        print "sol = "
        x = x[:nVar]
        #print x
        G = G.todense()
        print 0.5 * x * (x * G).T + np.dot(c, x) - self.objectiveOffset
        return

        print 'feasibility'
        print getSolution(s, 'z')
        print getSolution(s, 'y_A')
        #print (s.getPrimalSolution()[:nVar] * G).T
        #print '*****'
        #print  (s.getPrimalSolution()[:nVar] * G).T + getSolution(s, 's^+') -\
        #               getSolution(s, 's^-') -
        #               (getSolution(s, 'y_A') * A)[:nVar] - \
        #               getSolution(s, 'z')[:nVar] #+ np.dot(c, x)
        print '-c'
        print -c
        print getSolution(s, 'z')
        #print getSolution(s, 'y_A') * A + getSolution(s, 'z')
        print 's^+'
        print getSolution(s, 's^+')
        print 's^-'
        print getSolution(s, 's^-')

        x = s.getPrimalVariableSolution()
        print "sol = "
        print x
        G = G.todense()
        print G.shape
        print x * G
        print c
        print x
        print 0.5 * x * (x * G).T + np.dot(c, x)


def QPTest():
    qp = QP()
    start = clock()
    qp.fromQps(sys.argv[1])
    r = clock() - start
    qp.Wolfe()
    print 'took %g seconds to read the problem' % r
    print 'took %g seconds to solve the problem' % (clock() - start)
    print "done"

import sys
if __name__ == '__main__':
    QPTest()
    #cProfile.run('QPTest()')
