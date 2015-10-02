from __future__ import print_function
import numpy as np
from scipy import sparse
from cylp.cy import CyClpSimplex
from cylp.py.utils.sparseUtil import csr_matrixPlus
from cylp.py.modeling.CyLPModel import CyLPArray
from cylp.py.QP.QP import QP

class setCover:
    def __init__(self):
        self.nRows = self.nCols = 0
        self.cols = []
        self.costs = []

    def readWedelin(self, filename):
        '''
        Read a file describing a set-covering problem:
        #rows #ncols
        for each col:
            cost, #1's, list of 1 rows

        '''
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            tokens = lines[0].split()
            self.nRows = int(tokens[0])
            self.nCols = int(tokens[1])
            self.cols = []
            self.costs = []
            for line in lines[1:]:
                tokens = line.split()
                self.cols.append([int(i)-1 for i in tokens[2:]])
                self.costs.append(float(tokens[0]))

        print(self.nRows, self.nCols)

    def readBalas(self, filename):
        '''
        Read a file describing a set-covering problem:
        #ncols #nRows
        column costs
        for each col:
            #1's, list of 1 rows

        '''
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            tokens = lines[0].split()
            self.nRows = int(tokens[1])
            self.nCols = int(tokens[0])

            i = 1
            self.costs = []

            while len(self.costs) != self.nCols:
                self.costs += [float(k) for k in lines[i].split()]
                i += 1


            self.cols = []
            j = i
            while len(self.cols) != self.nCols:
                self.cols.append([])
                tokens = lines[j].split()
                nnz = int(tokens[0])
                tokens = tokens[1:]
                self.cols[-1] += [int(i)-1 for i in tokens]
                j += 1
                while len(self.cols[-1]) != nnz:
                    self.cols[-1] += [int(i)-1 for i in lines[j].split()]
                    j += 1
        print(self.nRows, self.nCols)


    @property
    def model(self):
        A = self.A
        c = self.c
        s = CyClpSimplex()

        x = s.addVariable('x', self.nCols)

        s += A * x >= 1
        s += 0 <= x <= 1

        s.objective = c * x

        return s


    def QPModel(self, addW=False):
        A = self.A
        c = self.c
        s = CyClpSimplex()

        x = s.addVariable('x', self.nCols)
        if addW:
            w = s.addVariable('w', self.nCols)

        s += A * x >= 1
        n = self.nCols

        if not addW:
            s += 0 <= x <= 1
        else:
            s += x + w == 1
            s += 0 <= w <= 1

##        s += -1 <= x <= 1

        s.objective = c * x

        if addW:
            G = sparse.lil_matrix((2*n, 2*n))
            for i in range(n/2, n): #range(n-1):
                G[i, i] = 1
            G[2*n-1, 2*n-1] = 10**-10
        else:
            G = sparse.lil_matrix((n, n))
            for i in range(n/2, n): #range(n-1):
                G[i, i] = 1


        s.Hessian = G
        return s

    @property
    def A(self):
        a = sparse.lil_matrix((self.nRows, self.nCols))
        for nCol in range(self.nCols):
            for nRow in self.cols[nCol]:
                a[nRow, nCol] = 1

        return csr_matrixPlus(a)

    @property
    def c(self):
        return CyLPArray(self.costs)
##        c = np.empty((self.nCols,), np.double)
##        print(self.nRows, self.nCols)
##        for nCol in range(self.nCols):
##            c[nCol] = self.costs[nCol]
##
##        return c

    @property
    def G(self):
        n = self.nCols

##        qp = QP()
##        qp.fromQps('/Users/mehdi/Documents/work/benchmarks/qp/CVXQP3_M.SIF')
##        #mm = qp.G[:n, :n].todense()
##        #print((mm == mm.T).all())
##        #l = np.linalg.eigvals(mm)
##        #print(min(l))
##        G = sparse.lil_matrix((2*n, 2*n))
####        G[n/2:n, n/2:n] = qp.G[n/2:n, n/2:n]
####        k = 10
####        G[n-k:n, n-k:n] = qp.G[:k, :k]
##        dim = min(qp.G.shape[0], n)
##        G[:dim, :dim] = qp.G[:dim, :dim]
##        if G[2*n-1, 2*n-1] == 0:
##            G[2*n-1, 2*n-1] = 10**-10
##        return G

        #n *= 2
        G = sparse.lil_matrix((2*n, 2*n))
        for i in range(n/2, n): #range(n-1):
            G[i, i] = 1
            #G[i+1, i] = -0.2
            #G[i, i+1] = -0.2
        #G[n - 1, n - 1] = 1
        G[2*n-1, 2*n-1] = 10**-10
        return csr_matrixPlus(G)


import sys




s = setCover()
#s.readWedelin(sys.argv[1])
s.readBalas(sys.argv[1])

of = sys.argv[2]
##f = filename[(filename.rindex('/') + 1): (filename.rindex('.'))] + '_X.qps'
##m = s.QPModel(False)
##m.writeMps('/Users/mehdi/temp/scp/' + f)
#f = filename[(filename.rindex('/') + 1): (filename.rindex('.'))] + '_W+.qps'

m = s.model
m.writeMps(of)


#m.primal()
#sol = m.primalVariableSolution['x']
#print([s.cols[i] for i in range(s.nCols) if sol[i] == 1])
