import numpy as np
from scipy import sparse
from CyLP.cy import CyClpSimplex
from CyLP.py.utils.sparseUtil import csr_matrixPlus


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


    @property
    def QPModel(self):
        A = self.A
        c = self.c
        s = CyClpSimplex()
        
        x = s.addVariable('x', self.nCols)

        s += A * x >= 1
        #s += 0 <= x <= 1
        
        s.objective = c * x
        
        s.Hessian = self.G
        return s
         
    @property
    def A(self):
        a = sparse.lil_matrix((self.nRows, self.nCols))
        for nCol in xrange(self.nCols):
            for nRow in self.cols[nCol]:
                a[nRow, nCol] = 1
            
        return csr_matrixPlus(a)
    
    @property
    def c(self):
        c = np.zeros(self.nCols, np.double)
        print self.nRows, self.nCols
        for nCol in xrange(self.nCols):
            c[nCol] = self.costs[nCol]
            
        return c

    @property
    def G(self):
        n = self.nCols
        G = sparse.lil_matrix((n, n))
        for i in xrange(n/2): #xrange(n-1):
            G[i, i] = 1
            #G[i+1, i] = -0.2
            #G[i, i+1] = -0.2
        G[n - 1, n - 1] = 1
        return G



s = setCover()
s.readWedelin('/Users/mehdi/Downloads/b727.dat')
#s.readBalas('/Users/mehdi/Downloads/bus1.inp')

m = s.QPModel
m.writeMps('/Users/mehdi/Desktop/b727_I05.mps')

#m.primal()
#sol = m.primalVariableSolution['x']
#print [s.cols[i] for i in range(s.nCols) if sol[i] == 1]
