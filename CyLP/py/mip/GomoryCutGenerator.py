import math
import numpy as np
from CyLP.cy import CyClpSimplex
from CyLP.py.modeling import CyLPArray, CyLPModel
from CyLP.py.modeling.CyLPModel import getCoinInfinity

epsilon = 0.01

def isInt(x):
    '''
    Return True if x is an integer, or if x is a numpy array
    with all integer elements, False otherwise
    '''
    if isinstance(x, (int, long, float)):
        return abs(math.floor(x) - x) < epsilon
    return (np.abs(np.floor(x) - x) < epsilon).all()

def gomoryCut(lp, rowInd):
    'Return the Gomory cut of row ``rowInd`` of lp (a CyClpSimplex object)'
    fractions = np.array([getFraction(x) for x in lp.tableau[rowInd, :]])
    pi, pi0 = fractions[:lp.nVariables], getFraction(lp.rhs[rowInd])
    pi_slacks = fractions[lp.nVariables:]
    #print 'g4'
    #print pi.shape
    #print pi_slacks.shape
    #print lp.coefMatrix.shape
    pi -= pi_slacks * lp.coefMatrix
    #print 'g5'
    pi0 -= np.dot(pi_slacks, lp.constraintsUpper)
    return pi, pi0

def getFraction(x):
    'Return the fraction part of x: x - floor(x)'
    return x - math.floor(x)

class GomoryCutGenerator:
    def __init__(self, cyLPModel):
        self.cyLPModel = cyLPModel

    def generateCuts(self, si, cglTreeInfo):
        #print '----------------------'
        #print '1'
        m = self.cyLPModel
        x = m.getVarByName('x')

        clpModel = si.clpModel
        #print 'solving'
        clpModel.dual(startFinishOptions='x')
        #print clpModel.tableau
        #print 'solved'
        solution = clpModel.primalVariableSolution
        bv = clpModel.basicVariables
        rhs = clpModel.rhs
        #print clpModel.dualVariableSolution

        intInds = clpModel.integerInformation

        rhsIsInt = map(isInt, rhs)

        #print '2'

        #print 'basic vars: ', bv
        #print 'rhs: ', rhs
        #print 'sol: ', solution
        #print 'objective: ', clpModel.objectiveValue

        cuts = []
        for rowInd in xrange(s.nConstraints):
            #print '2.1'
            basicVarInd = bv[rowInd]
            #print 'basicVarInd = ', basicVarInd
            #print 'intInds = ', intInds
            #print 'rhsIsInt = ', rhsIsInt

            if basicVarInd < clpModel.nVariables and intInds[basicVarInd] and not rhsIsInt[rowInd]:
                #print '2.2'
                coef, b = gomoryCut(clpModel, rowInd)
                #print '2.3'
                #print (abs(coef) <= 0.0001).all()
                if b > -1000:
                    print 'Adding cut: ', coef, '>=', b
                    cuts.append(CyLPArray(coef) * x >= b)
        #print '3'
        #print clpModel.isInteger(0)
        #print clpModel.isInteger(1)
        #print 'returning cuts'
        return cuts


if __name__ == '__main__':
    m = CyLPModel()

    if (False):
        x = m.addVariable('x', 2, isInt=True)

        A = np.matrix([[7., -2.],[0., 1], [2., -2]])
        b = CyLPArray([14, 3, 3])

        m += A * x <= b
        m += x >= 0

        c = CyLPArray([-4, 1])
        m.objective = c * x
        s = CyClpSimplex(m)
    else:
        s = CyClpSimplex()
        #m = s.extractCyLPModel('/Users/mehdi/Documents/CyLP/CyLP/input/netlib/boeing1.mps')
        m = s.extractCyLPModel('/Users/mehdi/Downloads/timtab1.mps')
        x = m.getVarByName('x')
        s.setInteger(x[:10])

    cbcModel = s.getCbcModel()

    gc = GomoryCutGenerator(m)
    cbcModel.addPythonCutGenerator(gc, name='PyGomory')

    cbcModel.branchAndBound()

    print cbcModel.primalVariableSolution

