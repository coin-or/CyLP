#UNDER DEVELOPMENT AND TEST

from __future__ import print_function
import sys
import cProfile
import inspect
from time import clock
import numpy as np
from scipy import sparse
from cylp.cy import CyClpSimplex
from cylp.py.QP.QPSReader import readQPS
from cylp.py.pivots.WolfePivot import WolfePivot
from cylp.py.pivots.WolfePivotPE import WolfePivotPE
from cylp.py.pivots.PositiveEdgeWolfePivot import PositiveEdgeWolfePivot
from cylp.cy import CyCoinModel
from cylp.py.utils.sparseUtil import I, sparseConcat
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray, CyLPVar

def getSolution(s, varGroupname):
    sol = s.getPrimalVariableSolution()
    return np.array([sol[i] for i in IndexFactory.varIndex[varGroupname]])


#def I(n):
#    '''
#    Return a sparse identity matrix of size *n*
#    '''
#    if n <= 0:
#        return None
#    return csc_matrixPlus(sparse.eye(n, n))

def checkComp(x, y):
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            return i
    return -1

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
        :type i: integer
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
        :arg x: vector
        :type x: Numpy array
        '''
        return self.G * x + self.a

    def fromQps(self, filename):
        (self.G, self.c, self.A, self.b,
            self.C, self.c_low, self.c_up,
            self.x_low, self.x_up,
            self.n, self.nEquality, self.nInEquality,
            self.objectiveOffset) = readQPS(filename)
        self.filename = filename
        # Use this later if slack variable are added
        self.nOriginalVar = self.n

    def convertToEqualityOnly(self, varsToo=True):
        '''
        Add necessary slack and surplus variables and return the
        Augmented ``A`` and ``b``.
        '''
        A = self.A
        G = self.G
        b = CyLPArray(self.b)
        c = self.c
        C = self.C
        c_low = CyLPArray(self.c_low)
        c_up = CyLPArray(self.c_up)
        x_low = self.x_low
        x_up = self.x_up
        infinity = self.infinity

        nVar = self.n
        nEquality = self.nEquality
        nInEquality = self.nInEquality

#        if nEquality:
#            print('A')
#            print(A.todense())
#            print(b)
#
#        if nInEquality:
#            print(C')
#            print(C.todense())
#            print(c_low)
#            print(c_up)
#
#        print('Hessian')
#        print(G.todense())
#
#        print('c')
#        print(c)

        iVarsWithJustUpperBound = [i for i in range(nVar)
                            if x_up[i] < infinity and x_low[i] <= -infinity]
        nVarsWithJustUpperBound = len(iVarsWithJustUpperBound)

        iVarsWithJustLowerBound = [i for i in range(nVar)
                            if x_low[i] > -infinity and x_up[i] >= infinity]
        nVarsWithJustLowerBound = len(iVarsWithJustLowerBound)

        iVarsWithBothBounds = \
                [i for i in range(nVar)
                        if x_low[i] > -infinity and x_up[i] < infinity]
        nVarsWithBothBounds = len(iVarsWithBothBounds)

        iFreeVars = \
                [i for i in range(nVar)
                        if x_low[i] <= -infinity and x_up[i] >= infinity]
        nFreeVars = len(iFreeVars)

        iConstraintsWithBothBounds = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] > -infinity]
        nConstraintsWithBothBounds = len(iConstraintsWithBothBounds)

        iConstraintsWithJustUpperBound = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] <= -infinity]
        nConstraintsWithJustUpperBound = len(iConstraintsWithJustUpperBound)

        iConstraintsWithJustLowerBound = \
                [i for i in range(nInEquality)
                        if c_up[i] >= infinity and c_low[i] > -infinity]
        nConstraintsWithJustLowerBound = len(iConstraintsWithJustLowerBound)

#        print('<=C<=', len(iConstraintsWithBothBounds))
#        print('C<=', len(iConstraintsWithJustUpperBound))
#        print('<=C', len(iConstraintsWithJustLowerBound))
#        print('<=x<=', len(iVarsWithBothBounds))
#        print('x<=', len(iVarsWithJustUpperBound))
#        print('<=x', len(iVarsWithJustLowerBound))
#        print('free x', len(iFreeVars))
        #st = 'n, m, =, <c<, c<, <c, <x<, x<, <x, freeVar\n'


#        st = ''
#        ind = self.filename.rindex('/') + 1
#        st += self.filename[ind:] + ', '
#        st +=  '%g, ' % nVar
#        st +=  '%g, ' % (nEquality + nInEquality)
#        st +=  '%g, '% nEquality
#        st +=  '%g, '% len(iConstraintsWithBothBounds)
#        st +=  '%g, '% len(iConstraintsWithJustUpperBound)
#        st += '%g, '% len(iConstraintsWithJustLowerBound)
#        st += '%g, '% len(iVarsWithBothBounds)
#        st += '%g, '% len(iVarsWithJustUpperBound)
#        st += '%g, '% len(iVarsWithJustLowerBound)
#        st += '%g, '% len(iFreeVars)
#
#        st += '%g, ' % (G.nnz / float(nVar*nVar))
#        st += '%g, ' % (A.nnz / float(A.shape[0] * A.shape[1]))

        if nInEquality:
            iden = I(nInEquality)
            for i in iConstraintsWithJustLowerBound:
                iden[i, i] = -1
            C = sparseConcat(C, iden, 'h')

            c_rhs = np.zeros(nInEquality)

            if nConstraintsWithBothBounds:
                c_rhs[iConstraintsWithBothBounds] = \
                                    c_up[iConstraintsWithBothBounds]
                c_rhs = np.concatenate((c_rhs, c_up[iConstraintsWithBothBounds] -
                                    c_low[iConstraintsWithBothBounds]), axis=0)

                C = sparseConcat(C, iden[iConstraintsWithBothBounds, :], 'v', h_offset=-1)

                C = sparseConcat(C, I(nConstraintsWithBothBounds), 'h', v_offset=-1)

            if nConstraintsWithJustUpperBound:
                c_rhs[iConstraintsWithJustUpperBound] = \
                                    c_up[iConstraintsWithJustUpperBound]

            if nConstraintsWithJustLowerBound:
                c_rhs[iConstraintsWithJustLowerBound] = \
                                    c_low[iConstraintsWithJustLowerBound]
        else:
            C = None
            c_rhs = np.array([])

        if varsToo:
            V = None
            iNonFree = np.sort(np.concatenate((iVarsWithBothBounds,
                                       iVarsWithJustUpperBound,
                                       iVarsWithJustLowerBound), axis=0))
            nNonFree = len(iNonFree)
            iden = I(nVar)

            for i in iVarsWithJustLowerBound:
                iden[i, i] = -1

            if nNonFree:
                V = sparseConcat(V, I(nVar)[iNonFree, :], 'h')
                V = sparseConcat(V, iden[iNonFree, :][:, iNonFree],
                             'h', h_offset=nInEquality +
                                                nConstraintsWithBothBounds)


            v_rhs = np.zeros(nVar)

            if nVarsWithBothBounds:
                v_rhs[iVarsWithBothBounds] = \
                                    x_up[iVarsWithBothBounds]
                v_rhs = np.concatenate((v_rhs, x_up[iVarsWithBothBounds] -
                                    x_low[iVarsWithBothBounds]), axis=0)

                V = sparseConcat(V, iden[iVarsWithBothBounds, :][:, iNonFree],
                             'v', h_offset=-1)

                V = sparseConcat(V, I(nVarsWithBothBounds), 'h', v_offset=-1)

            if nVarsWithJustUpperBound:
                v_rhs[iVarsWithJustUpperBound] = \
                                    x_up[iVarsWithJustUpperBound]

            if nVarsWithJustLowerBound:
                v_rhs[iVarsWithJustLowerBound] = \
                                    x_low[iVarsWithJustLowerBound]

            if nFreeVars:
                v_rhs = np.delete(v_rhs, iFreeVars, 0)

            c_rhs = np.concatenate((c_rhs, v_rhs), axis=0)
            C = sparseConcat(C, V, 'v')


        if nEquality:
            C = sparseConcat(A, C, 'v')
            c_rhs = np.concatenate((b, c_rhs), axis=0)


#        mmm = C.todense()
#        from math import ceil
#        secSize = 25
#        cols = mmm.shape[1]
#        divs = int(ceil(cols / float(secSize)))
#
#        for sec in range(divs):
#            for i in range(mmm.shape[0]):
#                for j in range(sec * secSize, min((sec+1) * secSize, mmm.shape[1])):
#                    print(str(int(round(mmm[i, j], 1))).rjust(4),)
#                print()
#            print('...')
#
#        print(c_rhs)

        assert(C.shape[0] == len(c_rhs))
        self.A = C
        self.b = c_rhs



        self.C = None
        self.c_low = None
        self.c_up = None
        self.n = self.A.shape[1]
        # Resize the Hessian to match the new self.n
        self.G[self.n - 1, self.n - 1] = 0
        self.nEquality = self.A.shape[0]
        self.nInEquality = 0

        self.x_low = -infinity * np.ones(self.n)
        self.x_low[nVar:] = 0
        self.x_up = infinity * np.ones(self.n)

#        st += '%g, ' % (self.G.nnz / float(self.G.shape[0]*self.G.shape[0]))
#        st += '%g ' % (self.A.nnz /
#                            float(self.A.shape[0] * self.A.shape[1]))
#        st += '\n'
#        with open('qpstat', 'a') as f:
#            f.write(st)


    def WolfeEquality(self, method='w'):
        assert(self.nInEquality == 0)
        start = clock()
        A = self.A
        b = CyLPArray(self.b)
        c = CyLPArray(self.c)
        G = self.G

#        print('A\n', A.todense())
#        print('b\n', b)
#        print('c\n', c)
#        print('G\n', G.todense())
#


        nVar = self.n
        nx = self.nOriginalVar
        nSlacks = nVar - nx


        m = CyLPModel()
        x = m.addVariable('x', nVar)
        m += A * x == b

        s = CyClpSimplex(m)

        if s.primal() != 'optimal':
            return



        sp = s.addVariable('sp', nVar)
        sm = s.addVariable('sm', nVar)
        z = s.addVariable('z', nSlacks)
        y = s.addVariable('y', self.nEquality)

        #from numpy import linalg as LA
        #print('cond:', LA.cond(G[:nx, :nx].todense()))
        #print('G nnz:', G.nnz, G.nnz / float(nx*nx))
        #print('A nnz:', A.nnz, A.nnz / float(A.shape[0] * A.shape[1]))

        #print(G.todense())
        minDiag = min(G[i, i] for i in range(nx))
        delta =  max(10**-8, 0.01 * max(1.0**-4, minDiag))
        G = G + delta * I(nVar)
        #print(G.todense())

#        print('G')
#        print(G.todense())
#
#        print('A')
#        print(A.todense())
#
#        print('A.T')
#        print(A.T.todense())
#
#        print('A.T[:nx, :]')
#        print(A.T[:nx, :].todense())
#
#        print('A.T[nx:, :]')
#        print(A.T[nx:, :].todense())
#
#        print(nx)

        s += G[:nx, :] * x - A.T[:nx, :] * y + sp[:nx] - sm[:nx] == -c[:nx]
        if nSlacks:
            s += -A.T[nx:, :] * y - z + sp[nx:] - sm[nx:] == 0
            s += z >= 0

        #s += A * x == b

        s += x[nx:] >= 0
        s += sp >= 0
        s += sm >= 0

        s.objective = sp.sum() + sm.sum()



        #s.writeMps('/Users/mehdi/Desktop/test.mps')

        p = WolfePivot(s)

        if method == 'wp':
            p = WolfePivotPE(s)

        if nSlacks:
            p.setComplement(m, x[nx:], z)
        #print('comp list:\n', p.complementarityList)

        s.setPivotMethod(p)
        timeToMake = clock() - start
        start = clock()
        s.primal()
        timeToSolve = clock() - start

        self.writeReport('qpout', s, timeToMake, timeToSolve, method, p)

        print('x')
        print(s.primalVariableSolution['x'])
#        print('z')
#        print(s.primalVariableSolution['z'])
#        print('y')
#        print(s.primalVariableSolution['y'])
#
#        nx = self.nOriginalVar
#        x = np.matrix(s.primalVariableSolution['x']).T
#        y = np.matrix(s.primalVariableSolution['y']).T
#        z = np.matrix(s.primalVariableSolution['z']).T
#        #x = x[:nx]
#        #G = self.G[:nx, :nx]
#        c = np.matrix(self.c).T
#        print(c)
#        print(np.zeros((3, 1))))
#        c = np.concatenate((c, np.zeros((5, 1))), axis=0)
#        print(c)
#
#        print('A*x = ')
#        print(A * x)
#        print('G * x + c')
#        print(G * x + c)
#        print('A.T * y')
#        print(A.T * y)
#
#        print(G[:nx, :] * x - A.T[:nx, :] * y)
#        print(-A.T[nx:, :] * y - z)
#        #print(G * x + c - A.T * y - z)
        return

#        m = CyLPModel()
#        x = m.addVariable('x', nVar)
#        sp = m.addVariable('sp', nVar)
#        sm = m.addVariable('sm', nVar)
#        z = m.addVariable('z', nSlacks)
#        y = m.addVariable('y', self.nEquality)
#
#        m += G[:nx, :] * x - A.T[:nx, :] * y + sp[:nx] - sm[:nx] == -c[:nx]
#        m += -A.T[nx:, :] * y - z + sp[nx:] - sm[nx:] == 0
#
#        m += A * x == b
#
#        m += x[nx:] >= 0
#        m += z >= 0
#        m += sp >= 0
#        m += sm >= 0
#        m += z >= 0
#
#        m.objective = sp.sum() + sm.sum()
#
#        s = CyClpSimplex(m)
#
#
#        #s.writeMps('/Users/mehdi/Desktop/test.mps')
#
#        p = WolfePivot(s)
#
#        if method == 'wp':
#            p = WolfePivotPE(s)
#
#        p.setComplement(m, x[nx:], z)
#        #print('comp list:\n', p.complementarityList)
#
#        s.setPivotMethod(p)
##        timeToMake = clock() - start
##        start = clock()
#        s.primal()



#        timeToSolve = clock() - start
        #s.initialPrimalSolve()
        if method == 'wp':
            total = p.compCount + p.nonCompCount
            print('comp : %g ' % (p.compCount / float(total)))
            print('comp rejection : %g' % (p.compRej / float(p.compCount)))
        #print(s.primalVariableSolution)
        print('OBJ:', s.objectiveValue)
        x = np.matrix(s.primalVariableSolution['x']).T
        #print(A * x - np.matrix(b).T)
        print('objective:')
        x = x[:nx]
        G = G[:nx, :nx]
        qobj = 0.5 * x.T * G * x + np.dot(c, x) - self.objectiveOffset
        print(qobj)
        print(s.primalVariableSolution)

#        print(s.iteration)
#        f = open('qpout', 'a')
#        st = '%s %s %s %s %s %s %s\n' % (self.filename.ljust(30), method.ljust(2),
#                str(round(s.objectiveValue, 5)).ljust(8),
#                str(round(qobj, 5)).ljust(8),
#                str(timeToMake),
#                str(timeToSolve),
#                str(timeToMake + timeToSolve))
#        f.write(st)
#        f.close()

    def writeReport(self, filename, s, timeToMake, timeToSolve, method, p):

        def getcell(s, r, j):
            return str(round(s, r)).ljust(j)

        nx = self.nOriginalVar
        x = np.matrix(s.primalVariableSolution['x']).T
        x = x[:nx]
        G = self.G[:nx, :nx]
        c = self.c

        qobj = 0.5 * x.T * G * x + np.dot(c, x) - self.objectiveOffset

        if method == 'wp':
            total = p.compCount + p.nonCompCount
            compPer = (p.compCount / float(total))
            compRejPer = (p.compRej / float(p.compCount))

            st = '%s %s %s %s %s %s %d %s %s\n' % (self.filename.ljust(30),
                method.ljust(2),
                getcell(s.objectiveValue, 5, 8),
                getcell(qobj, 5, 8),
                getcell(timeToMake, 3, 8),
                getcell(timeToSolve, 3, 8),
                s.iteration,
                getcell(compPer, 2, 5),
                getcell(compRejPer, 2, 5))
        else:
            st = '%s %s %s %s %s %s %d\n' % (self.filename.ljust(30), method.ljust(2),
                getcell(s.objectiveValue, 5, 8),
                getcell(qobj, 5, 8),
                getcell(timeToMake, 3, 8),
                getcell(timeToSolve, 3, 8),
                s.iteration)

        with open('qpout', 'a') as f:
            f.write(st)

    def Wolfe_2(self):
        '''
        Solve a QP using Wolfe's method
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
        nCons = nEquality + nInEquality

        varIndexDic = {}
        currentNumberOfVars = 0
        constIndexDic = {}
        currentNumberOfConst = 0

        s = CyClpSimplex()
        model = CyCoinModel()

        inds = range(nVar)

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
                [i for i in range(len(x_up)) if x_up[i] < infinity]
        nVarsWithUpperBound = len(iVarsWithUpperBound)

        iVarsWithLowerBound = \
                [i for i in range(len(x_low)) if x_low[i] > -infinity]
        nVarsWithLowerBound = len(iVarsWithLowerBound)

        iVarsWithBothBounds = \
                [i for i in range(len(x_low)) if
                 x_low[i] > -infinity and x_low[i] < infinity]
        nVarsWithBothBounds = len(iVarsWithBothBounds)

        iConstraintsWithBothBounds = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] > -infinity]
        nConstraintsWithBothBounds = len(iConstraintsWithBothBounds)

        iConstraintsWithJustUpperBound = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] <= -infinity]
        nConstraintsWithJustUpperBound = len(iConstraintsWithJustUpperBound)

        iConstraintsWithJustLowerBound = \
                [i for i in range(nInEquality)
                        if c_up[i] >= infinity and c_low[i] > -infinity]
        nConstraintsWithJustLowerBound = len(iConstraintsWithJustLowerBound)

        iVarsWithUpperBound = [i for i in inds if x_up[i] < infinity]
        nVarsWithUpperBound = len(iVarsWithUpperBound)

        iVarsWithLowerBound = [i for i in inds if x_low[i] > -infinity]
        nVarsWithLowerBound = len(iVarsWithLowerBound)

        return


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
            #print('rowi : ')
            #print(rowi.col)
            cols = rowi.col + IndexFactory.constIndex['Gx=-c'][0]
            #print(cols)
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
        print("CLP time : %g seconds" % (clock() - st))

        x = s.getPrimalVariableSolution()
        print("sol = ")
        x = x[:nVar]
        #print(x)
        G = G.todense()
        #print(G.shape)
        #print(x * G)
        #print(c)
        #print(x)
        print(0.5 * x * (x * G).T + np.dot(c, x) - self.objectiveOffset)

        return

    def Wolfe(self, method='w'):
        '''
        Solves a QP using Wolfe's method (``method = 'w'``) or Wolfe's method using
        positive edge as pivot rule (``method = 'wp'``).
        '''
        start = clock()
        A = self.A
        G = self.G
        b = CyLPArray(self.b)
        c = self.c
        C = self.C
        c_low = CyLPArray(self.c_low)
        c_up = CyLPArray(self.c_up)
        x_low = self.x_low
        x_up = self.x_up
        infinity = self.infinity

        nVar = self.n
        nEquality = self.nEquality
        nInEquality = self.nInEquality

#        print('A')
#        print(A)
#        print(b)
#
#        print('C')
#        print(C)
#        print(c_low)
#        print(c_up)
#        print('Hessian')
#        print(G)
#
#        print('c')
#        print(c)

        iVarsWithJustUpperBound = [i for i in range(nVar)
                            if x_up[i] < infinity and x_low[i] <= -infinity]
        nVarsWithJustUpperBound = len(iVarsWithJustUpperBound)

        iVarsWithJustLowerBound = [i for i in range(nVar)
                            if x_low[i] > -infinity and x_up[i] >= infinity]
        nVarsWithJustLowerBound = len(iVarsWithJustLowerBound)

        iVarsWithBothBounds = \
                [i for i in range(nVar)
                        if x_low[i] > -infinity and x_up[i] < infinity]
        nVarsWithBothBounds = len(iVarsWithBothBounds)

        iConstraintsWithBothBounds = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] > -infinity]
        nConstraintsWithBothBounds = len(iConstraintsWithBothBounds)

        iConstraintsWithJustUpperBound = \
                [i for i in range(nInEquality)
                        if c_up[i] < infinity and c_low[i] <= -infinity]
        nConstraintsWithJustUpperBound = len(iConstraintsWithJustUpperBound)

        iConstraintsWithJustLowerBound = \
                [i for i in range(nInEquality)
                        if c_up[i] >= infinity and c_low[i] > -infinity]
        nConstraintsWithJustLowerBound = len(iConstraintsWithJustLowerBound)

#        print('____________________________________________')
#        print(x_up)
#        print(x_low)
#
#        print(c_low)
#        print(c_up)
#
#        print(C.todense())
#        print(iConstraintsWithBothBounds)
#        print(iConstraintsWithJustLowerBound)
#        print(iConstraintsWithJustUpperBound)
#
#        print('vars')
#        print(iVarsWithBothBounds)
#        print(iVarsWithJustLowerBound)
#        print(iVarsWithJustUpperBound)
#        print('____________________________________________')


        In = I(nVar)

        m = CyLPModel()
        x = m.addVariable('x', nVar)

        yx1 = CyLPVar('yx1', dim=0)
        yxu = CyLPVar('yxu', dim=0)
        yx2 = CyLPVar('yx2', dim=0)
        yx3 = CyLPVar('yx3', dim=0)
        if nVarsWithBothBounds:
            k1 = m.addVariable('k1', nVarsWithBothBounds)
            zk1 = m.addVariable('zk1', nVarsWithBothBounds)
            spk1 = m.addVariable('spk1', nVarsWithBothBounds)
            smk1 = m.addVariable('smk1', nVarsWithBothBounds)
            yx1 = m.addVariable('yx1', nVarsWithBothBounds)
            m.addConstraint(k1 >= 0)
            m.addConstraint(spk1 >= 0)
            m.addConstraint(smk1 >= 0)
            m.addConstraint(zk1 >= 0)
            ku = m.addVariable('ku', nVarsWithBothBounds)
            zku = m.addVariable('zku', nVarsWithBothBounds)
            spku = m.addVariable('spku', nVarsWithBothBounds)
            smku = m.addVariable('smku', nVarsWithBothBounds)
            yxu = m.addVariable('yxu', nVarsWithBothBounds)
            m.addConstraint(ku >= 0)
            m.addConstraint(zku >= 0)
            m.addConstraint(spku >= 0)
            m.addConstraint(smku >= 0)
            m.addConstraint(x[iVarsWithBothBounds] + k1 ==
                            x_up[iVarsWithBothBounds], 'x1+k1')
            m.addConstraint(k1 + ku ==
                    (x_up[iVarsWithBothBounds] -
                                        x_low[iVarsWithBothBounds]), 'k1+ku')

        if nVarsWithJustUpperBound:
            k2 = m.addVariable('k2', nVarsWithJustUpperBound)
            zk2 = m.addVariable('zk2', nVarsWithJustUpperBound)
            #spk2 = m.addVariable('spk2', nVarsWithJustUpperBound)
            #smk2 = m.addVariable('smk2', nVarsWithJustUpperBound)
            yx2 = m.addVariable('yx2', nVarsWithJustUpperBound)
            m.addConstraint(k2 >= 0)
            m.addConstraint(zk2 >= 0)
            #m.addConstraint(spk2 >= 0)
            #m.addConstraint(smk2 >= 0)
            m.addConstraint(x[iVarsWithJustUpperBound] + k2 ==
                            x_up[iVarsWithJustUpperBound], 'x2+k2')

        if nVarsWithJustLowerBound:
            k3 = m.addVariable('k3', nVarsWithJustLowerBound)
            zk3 = m.addVariable('zk3', nVarsWithJustLowerBound)
            #spk3 = m.addVariable('spk3', nVarsWithJustLowerBound)
            #smk3 = m.addVariable('smk3', nVarsWithJustLowerBound)
            yx3 = m.addVariable('yx3', nVarsWithJustLowerBound)
            m.addConstraint(k3 >= 0)
            m.addConstraint(zk3 >= 0)
            #m.addConstraint(spk3 >= 0)
            #m.addConstraint(smk3 >= 0)
            m.addConstraint(x[iVarsWithJustLowerBound] -  k3 ==
                            x_low[iVarsWithJustLowerBound], 'x3-k3')
#            m.addConstraint(I(nVarsWithJustLowerBound) *
#                                    x[iVarsWithJustLowerBound] -
#                            I(nVarsWithJustLowerBound) * k3 ==
#                            x_low[iVarsWithJustLowerBound], 'x3-k3')

        if nEquality > 0 :
            m.addConstraint(A * x == b, 'Ax=b')
            yb = m.addVariable('yb', nEquality)

        C1T = C2T = C3T = None
        yc1 = CyLPVar('yc1', dim=0)
        ycu = CyLPVar('ycu', dim=0)
        yc2 = CyLPVar('yc2', dim=0)
        yc3 = CyLPVar('yc3', dim=0)
        if nConstraintsWithBothBounds:
            C1 = C[iConstraintsWithBothBounds, :]
            C1T = C1.T
            # g1: slack for ineq. const. with both bounds
            g1 = m.addVariable('g1', nConstraintsWithBothBounds)
            zg1 = m.addVariable('zg1', nConstraintsWithBothBounds)
            spg1 = m.addVariable('spg1', nConstraintsWithBothBounds)
            smg1 = m.addVariable('smg1', nConstraintsWithBothBounds)
            yc1 = m.addVariable('yc1', nConstraintsWithBothBounds)
            m.addConstraint(g1 >= 0)
            m.addConstraint(spg1 >= 0)
            m.addConstraint(smg1 >= 0)
            m.addConstraint(zg1 >= 0)
            # gu: slack of g1 from c_up - c_low
            gu = m.addVariable('gu', nConstraintsWithBothBounds)
            zgu = m.addVariable('zgu', nConstraintsWithBothBounds)
            spgu = m.addVariable('spgu', nConstraintsWithBothBounds)
            smgu = m.addVariable('smgu', nConstraintsWithBothBounds)
            ycu = m.addVariable('ycu', nConstraintsWithBothBounds)
            m.addConstraint(gu >= 0)
            m.addConstraint(zgu >= 0)
            m.addConstraint(spgu >= 0)
            m.addConstraint(smgu >= 0)
            m.addConstraint(C1 * x +
                            g1 == #+ spg1 - smg1 ==
                            c_up[iConstraintsWithBothBounds],
                            'C1x+g1')
            m.addConstraint(g1 + gu == #+ spgu - smgu ==
                            (c_up[iConstraintsWithBothBounds] -
                             c_low[iConstraintsWithBothBounds]),
                             'g1+gu')


        if nConstraintsWithJustUpperBound:
            C2 = C[iConstraintsWithJustUpperBound, :]
            C2T = C2.T
            g2 = m.addVariable('g2', nConstraintsWithJustUpperBound)
            zg2 = m.addVariable('zg2', nConstraintsWithJustUpperBound)
            #spg2 = m.addVariable('spg2', nConstraintsWithJustLowerBound)
            #smg2 = m.addVariable('smg2', nConstraintsWithJustLowerBound)
            yc2 = m.addVariable('yc2', nConstraintsWithJustUpperBound)
            m.addConstraint(g2 >= 0)
            m.addConstraint(zg2 >= 0)
            #m.addConstraint(spg2 >= 0)
            #m.addConstraint(smg2 >= 0)
            m.addConstraint(C2 * x + g2 == #+ spg2 - smg2 ==
                        c_up[iConstraintsWithJustUpperBound],
                                'C2x+g2')
        if nConstraintsWithJustLowerBound:
            C3 = C[iConstraintsWithJustLowerBound, :]
            C3T = C3.T
            g3 = m.addVariable('g3', nConstraintsWithJustLowerBound)
            zg3 = m.addVariable('zg3', nConstraintsWithJustLowerBound)
            spg3 = m.addVariable('spg3', nConstraintsWithJustLowerBound)
            smg3 = m.addVariable('smg3', nConstraintsWithJustLowerBound)
            yc3 = m.addVariable('yc3', nConstraintsWithJustLowerBound)
            m.addConstraint(g3 >= 0)
            m.addConstraint(zg3 >= 0)
            m.addConstraint(spg3 >= 0)
            m.addConstraint(smg3 >= 0)
            m.addConstraint(C3 * x -
                                g3 == #+ spg3 - smg3  ==
                                c_low[iConstraintsWithJustLowerBound],
                                'C3x-g3')


        x1CoefT = x2CoefT = x3CoefT = xuCoefT = None
        if nVarsWithBothBounds:
            x1CoefT = In[iVarsWithBothBounds, :].T
            xuCoefT = In[iVarsWithBothBounds, :].T
            Ik = I(nVarsWithBothBounds)
            k1CoefT = Ik
            kuCoefT = Ik


        if nVarsWithJustUpperBound:
            x2CoefT =  In[iVarsWithJustUpperBound, :].T
            Ik = I(nVarsWithJustUpperBound)
            k2CoefT =  Ik

        if nVarsWithJustLowerBound:
            x3CoefT = In[iVarsWithJustLowerBound, :].T
            Ik = I(nVarsWithJustLowerBound)
            k3CoefT = Ik

        sp = m.addVariable('sp', nVar)
        m.addConstraint(sp >= 0)
        sm = m.addVariable('sm', nVar)
        m.addConstraint(sm >= 0)

        #z = m.addVariable('z', nVar)


        # Dual-feasibility constraints:
        if nEquality:
            #from pudb import set_trace; set_trace()
            m.addConstraint(G * x - A.T * yb - C1T * yc1 - C2T * yc2 - C3T * yc3 -
                        x1CoefT * yx1 - x2CoefT * yx2 - x3CoefT * yx3 + sp - sm  == -c,
                        'Gx-ATy-CTu-z')
        else:
            m.addConstraint(G * x - C1T * yc1 - C2T * yc2 - C3T * yc3 -
                        x1CoefT * yx1 - x2CoefT * yx2 - x3CoefT * yx3 + sp - sm  == -c ,
                        'Gx-CTu-z')


        if nInEquality:
            #Im = I(nInEquality)

            if nConstraintsWithBothBounds:
                g1Coef = I(nConstraintsWithBothBounds)
                guCoef = I(nConstraintsWithBothBounds)
                #g1Coef = Im[:nConstraintsWithBothBounds, :]
                #guCoef = Im[:nConstraintsWithBothBounds, :]
                #g1Coef = Im[iConstraintsWithBothBounds, :]
                #guCoef = Im[iConstraintsWithBothBounds, :]
                #g1Coef = I(nConstraintsWithBothBounds)
                #guCoef = I(nConstraintsWithBothBounds)
                m.addConstraint(-g1Coef.T * yc1 - g1Coef.T * ycu - zg1 +
                                spg1 - smg1 == 0,
                                'dualfeas_g1')
                m.addConstraint(-guCoef.T * ycu - zgu +
                                 spgu - smgu == 0,
                                'dualfeas_gu')

            if nConstraintsWithJustUpperBound:
                g2Coef = I(nConstraintsWithJustUpperBound)
                #g2Coef = Im[nConstraintsWithBothBounds :
                #             (nConstraintsWithBothBounds +
                #                 nConstraintsWithJustUpperBound), :]
                #g2Coef = Im[iConstraintsWithJustUpperBound, :]
                #g2Coef = I(nConstraintsWithJustUpperBound)
                m.addConstraint(-g2Coef.T * yc2 - zg2 == 0, 'dualfeas_g2')

            if nConstraintsWithJustLowerBound:
                g3Coef = I(nConstraintsWithJustLowerBound)
                #g3Coef = Im[(nConstraintsWithBothBounds +
                #             nConstraintsWithJustUpperBound):, :]
                #g3Coef = Im[iConstraintsWithJustLowerBound, :]
                #g3Coef = I(nConstraintsWithJustLowerBound)
                m.addConstraint(g3Coef.T * yc3 - zg3 + spg3 - smg3 == 0, 'dualfeas_g3')


        if nVarsWithBothBounds:
            m.addConstraint(-k1CoefT * yx1 - k1CoefT * yxu - zk1 + \
                            spk1 - smk1 == 0,
                            'dualfeas_k1')
            m.addConstraint(-kuCoefT * yxu - zku + spku - smku == 0, 'dualfeas_ku')

        if nVarsWithJustUpperBound:
            m.addConstraint(-k2CoefT * yx2 - zk2 == 0, 'dualfeas_k2')

        if nVarsWithJustLowerBound:
            #m.addConstraint(x3CoefT * yx3 - zk3 + spk3 - smk3 == 0, 'dualfeas_k3')
            m.addConstraint(k3CoefT * yx3 - zk3 == 0, 'dualfeas_k3')


#        m.objective =  sp.sum() + sm.sum() + spg1.sum() + smg1.sum() + \
#                        spgu.sum() + smgu.sum() + spg3.sum() + smg3.sum() + \
#                        spk1.sum() + smk1.sum() + spku.sum() + smku.sum()
        m.objective =  sp.sum() + sm.sum() + spg1.sum() + smg1.sum() + \
                        spgu.sum() + smgu.sum()
        #z = m.addVariable('z', nVar)

        s = CyClpSimplex(m)
        ###s.setComplement(x, z)
        #print(m.inds)
        #s.writeMps('/Users/mehdi/Desktop/test.mps')
        #s.useCustomPrimal(True)


        p = WolfePivot(s)

        if method == 'wp':
            p = WolfePivotPE(s)

        if nConstraintsWithBothBounds:
            p.setComplement(m, g1, zg1)
            p.setComplement(m, gu, zgu)

        if nConstraintsWithJustUpperBound:
            p.setComplement(m, g2, zg2)

        if nConstraintsWithJustLowerBound:
            p.setComplement(m, g3, zg3)
            #p.setComplement(m, sc3, yc3)


        if nVarsWithBothBounds:
            p.setComplement(m, k1, zk1)
            p.setComplement(m, ku, zku)

        if nVarsWithJustUpperBound:
            p.setComplement(m, k2, zk2)

        if nVarsWithJustLowerBound:
            p.setComplement(m, k3, zk3)

        #print(p.complementarityList)
        s.setPivotMethod(p)
        timeToMake = clock() - start
        start = clock()
        s.primal()
        timeToSolve = clock() - start
        #s.initialPrimalSolve()
        if method == 'wp':
            total = p.compCount + p.nonCompCount
            print('comp : %g ' % (p.compCount / float(total)))
            print('comp rejection : %g' % (p.compRej / float(total)))
        #print(s.primalVariableSolution)
        print('OBJ:', s.objectiveValue)
        x = np.matrix(s.primalVariableSolution['x']).T
        print('objective:')
        qobj = 0.5 * x.T * G * x + np.dot(c, x) - self.objectiveOffset

        print(s.iteration)
        f = open('qpout', 'a')
        st = '%s %s %s %s %s %s %s\n' % (self.filename.ljust(30), method.ljust(2),
                str(round(s.objectiveValue, 5)).ljust(8),
                str(round(qobj, 5)).ljust(8),
                str(timeToMake),
                str(timeToSolve),
                str(timeToMake + timeToSolve))
        f.write(st)
        f.close()

#        print(checkComp(s.primalVariableSolution['k1'],
#                        s.primalVariableSolution['zk1']))
#
#        print(checkComp(s.primalVariableSolution['ku'],
#                        s.primalVariableSolution['zku']))

#        print('A:')
#        print(A)
#        print('x')
#        print(x)
#        print('Ax:')
#        print(A * x)
#        print('b')
#        print(b)
#        y = np.matrix(s.primalVariableSolution['yb']).T
#        #yx1 = s.primalVariableSolution['yx1']
#
#        print(A.shape)
#        print(((A * x).T - b < 10 ** -5).all())
#
#        print('opt')
#        print(G.todense())
#        print(np.linalg.eigvals(G.todense()))
#        print(G * x - A.T * y + np.matrix(c).T)
#        x = s.primalVariableSolution['x']
#
#        for i in range(nVar):
#            if x[i] > x_up[i] + 10**-10 or x[i] < x_low[i] - 10**-10:
#                print(i, 'INFEASIBLE', x_low[i] , x[i], x_up[i], x[i] > x_up[i], x[i] < x_low[i])

        #print('Cx')
        #print(C * x)
        #print('g3')
        #print(s.primalVariableSolution['g3'])
        #print('Cx - g3')
        #print(C * x - np.matrix(s.primalVariableSolution['g3']).T)
        #print('optimality:')
        #print(-c)
        #y =  s.primalVariableSolution['yc3']
        #print(G*x)
        #print(C.T * y)
        #print('opt:')
        #print((G * x).T - C.T * y + c)

#        print('-------------------------------------------------------')
#        x = np.matrix([[1, 2, -1, 3, -4]]).T
#        print(0.5 * x.T * G * x + np.dot(c, x) - self.objectiveOffset
#        print('Cx')
#        print(C * x)
#        print('Cx - c_low')
#        print(C * x - np.matrix(c_low).T)


        return
        m.addConstraint(z >= 0)

        C = C.todense()
        G = G.todense()
        if nEquality > 0 :
            m.addConstraint(A * x == b)
            y = m.addVariable('y', nEquality)

        if nInEquality > 0 :
            c_low = CyLPArray(c_low)
            c_up = CyLPArray(c_up)
            print(C)
            #c_low *= -1
            #m.addConstraint(c_low <= C * x <= c_up)
            #ss = m.addVariable('ss', 1)
            m.addConstraint(-C * x  == -c_low)
            #m.addConstraint(-C * x <= -c_low)

            u = m.addVariable('u', nInEquality)
            #sp = m.addVariable('sp', nVar)
            #sm = m.addVariable('sm', nVar)
            #s = m.addVariable('s', 1)
            #m.addConstraint(ss >= 0)
            #m.addConstraint(sp >= 0)
            #m.addConstraint(sm >= 0)
            m.addConstraint(u >= 0)
            #m.addConstraint(s >= 0)
            #m.addConstraint(C * x  - s == c_low)


        if nEquality > 0:
            if nInEquality == 0:
                m.addConstraint(G * x - A.T * y - z == -c)
            else:
                m.addConstraint(G * x - A.T * y - C.T * u - z == -c)
        elif nInEquality > 0:
#            print(G.shape)
#            print(x.dim)
#            print(C.T)
#            print(u.dim)
#            print(c.shape)
#            print(z.dim)
            m.addConstraint(G * x - z - C.T * u + sp - sm == -c)  #- C.T * u - z

        m.objective = sm + sp

        s = CyClpSimplex(m)
        return
        #s.setComplement(x, z)
        s.useCustomPrimal(True)

        p = WolfePivot(s)
        s.setPivotMethod(p)

        s.primal()
        #s.initialPrimalSolve()
        print(s.primalVariableSolution)
        print(s.objectiveValue)
        return

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
            #print('rowi : ')
            #print(rowi.col)
            cols = rowi.col + IndexFactory.constIndex['Gx=-c'][0]
            #print(cols)
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
        print("CLP time : %g seconds" % (clock() - st))

        x = s.getPrimalVariableSolution()
        print("sol = ")
        x = x[:nVar]
        #print(x)
        G = G.todense()
        print(0.5 * x * (x * G).T + np.dot(c, x) - self.objectiveOffset)
        return

        print('feasibility')
        print(getSolution(s, 'z'))
        print(getSolution(s, 'y_A'))
        #print((s.getPrimalSolution()[:nVar] * G).T)
        #print('*****')
        #print((s.getPrimalSolution()[:nVar] * G).T + getSolution(s, 's^+') -\
        #               getSolution(s, 's^-') -
        #               (getSolution(s, 'y_A') * A)[:nVar] - \
        #               getSolution(s, 'z')[:nVar] #+ np.dot(c, x))
        print('-c')
        print(-c)
        print(getSolution(s, 'z'))
        #print(getSolution(s, 'y_A') * A + getSolution(s, 'z'))
        print('s^+')
        print(getSolution(s, 's^+'))
        print('s^-')
        print(getSolution(s, 's^-'))

        x = s.getPrimalVariableSolution()
        print("sol = ")
        print(x)
        G = G.todense()
        print(G.shape)
        print(x * G)
        print(c)
        print(x)
        print(0.5 * x * (x * G).T + np.dot(c, x))


def getStat():
    import os
    d = '/Users/mehdi/Documents/work/benchmarks/qp/'
    for f in os.listdir(d):
        if f[-3:] == 'SIF':
            qp = QP()
            qp.fromQps(d + f)
            qp.convertToEqualityOnly()


def QPTest():
    qp = QP()
    start = clock()
    qp.fromQps(sys.argv[1])
    qp.convertToEqualityOnly()
    if len(sys.argv) > 2:
        qp.WolfeEquality(sys.argv[2])
    else:
        qp.WolfeEquality()
    return
    r = clock() - start
    if len(sys.argv) > 2:
        qp.Wolfe(sys.argv[2])
    else:
        qp.Wolfe()
    print('took %g seconds to read the problem' % r)
    print('took %g seconds to solve the problem' % (clock() - start))
    print("done")

import sys
if __name__ == '__main__':
    QPTest()
    #cProfile.run('QPTest()')
