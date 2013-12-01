#import random
import numpy as np
from numpy import random
from scipy import sparse
from cylp.cy import CyClpSimplex
from cylp.py.modeling import CyLPArray
from cylp.py.utils.sparseUtil import csr_matrixPlus
from cylp.py.QP.QP import QP

def getA(nRows, nCols, nnzPerCol):
    '''Return a sparse coef. matrix of a set-partitioning problem
    *nnzPerCol* specifies the number of non-zero elements in 
    each column.
    '''
    A = sparse.lil_matrix((nRows, nCols))
    solCols = []
    for nRow in xrange(nRows):
        nCol = random.randint(0, nCols-1)
        A[nRow, nCol] = 1
        solCols.append(nCol)
   
    for nCol in [j for j in xrange(nCols) if j not in solCols]:
        for i in xrange(nnzPerCol):
            #if random.randint(0, 1):
                A[random.randint(0, nRows-1), nCol] = 1
            #else:
            #    A[random.randint(0, nRows-1), nCol] = -1
        
    return csr_matrixPlus(A)

def getG(nCols):
    n = nCols
    G = sparse.lil_matrix((n, n))
    for i in xrange(0, n-1):
        G[i, i] = 1
        #G[i+1, i] = -0.2
        #G[i, i+1] = -0.2
    G[nCols - 1, nCols - 1] = 1
    
    return csr_matrixPlus(G)

def getCoinInfinity():
    return 1.79769313486e+308

def generateRandomPositiveDefiniteMatrix(n, cond):
        random.seed()
        p = n
        N_of_P = np.matrix(np.random.standard_normal((p,p)))
        N_of_Q = np.matrix(np.random.standard_normal((n,n)))
        P = np.linalg.qr(N_of_P)[0]
        Q = np.linalg.qr(N_of_Q)[0]
        
        D = np.zeros((p, n))
        t_prime = cond**0.25
        D[0,0] = 1.0/t_prime
        D[n-1,n-1] = t_prime 
        for i in range(1, n-1):
            D[i, i] = t_prime**(random.uniform(-1,1))
    
        C  = P * D * Q
        C = C / np.linalg.norm(C, 2)
        G = C.T * C
        G = 2 * G
        return G

def getQP(m, n):
    qp = QP()
    qp.G = csr_matrixPlus(generateRandomPositiveDefiniteMatrix(n, 10))  #getG(n)
    #qp.c = np.zeros(n)
    qp.c = np.random.random(n) * 50

    qp.C = getA(m, n, 25)
    qp.c_low = np.ones(m) * -getCoinInfinity() 
    qp.c_up = np.ones(m)
#    qp.c_low = -np.ones(m) 
#    qp.c_up = np.ones(m) * getCoinInfinity()
    #qp.c_up = (20 + np.random.random(m) * 40) * 10**-3
    qp.b = qp.A = 0
    qp.nInEquality = m
    
    #qp.C =qp.c_low = qp.c_up = 0  
    #qp.A = getA(m, n, 17)
    #qp.b = np.ones(m)
    qp.nEquality = 0

    qp.x_low = np.ones(n) * -getCoinInfinity()
    qp.x_up = np.ones(n) * getCoinInfinity()
    
    qp.n = n
    qp.nOriginalVar = n
    qp.objectiveOffset = 0
    qp.filename = '/Users/mehdi/Desktop/qptest.txt'
    return qp

class QPGen:
    def __init__(self, n, m, HesCond, nnzPerCol, costRange, signs):
        self.n = n
        self.m = m
        self.HesCond = HesCond
        self.nnzPerCol = nnzPerCol
        self.costRange = costRange
        self.signs = signs
        self.generateQP()

    def generateQP(self):
        m = self.m
        n = self.n
        s = CyClpSimplex()
       
        iNonZero = set(random.randint(n, size=self.nnzPerCol))
        iZero = [i for i in xrange(n) if i not in iNonZero]
        x_star = np.matrix(np.zeros((n, 1)))
        z_star = np.matrix(np.zeros((n, 1)))
        
        for i in iNonZero:
            x_star[i, 0] = 1
        
        for i in iZero:
            z_star[i, 0] = 0 if random.randint(2) else random.random()

        G = getG(n)
        
        A = getA(m, n, self.nnzPerCol)
        y_star = np.matrix(random.random((m, 1)))
      
        c = -(G * x_star - A.T * y_star - z_star)
        
        obj = 0.5 * ((x_star.T * G) * x_star) + c.T * x_star
        print obj 

        c = CyLPArray((c.T)[0])
        
        b = CyLPArray(((A * x_star).T)[0])
        b = np.squeeze(np.asarray(b))

        x = s.addVariable('x', n)
 
        s += A * x == b
        s += x >= 0

        
        c = CyLPArray(c) 
        s.objective = c * x
        
        s.Hessian = G
        
        self.model = s
        return s

    def writeToFile(self):
        filename = 'qp_%d_%d_%d_%d_%d_%s.qps' % (self.n, self.m, self.HesCond, 
                                   self.nnzPerCol, self.costRange, self.signs)
        filename = 'tests/' + filename
        print 'writing to %s.' % filename
        self.model.writeMps(filename)



if __name__ == '__main__':
    import cProfile
    import sys
    n, m = int(sys.argv[1]), int(sys.argv[2])
    print m
    print m/10
    qp = QPGen(n, m, 1, m/10, 10000, 'G')
    qp.writeToFile()
    
