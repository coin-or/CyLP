import os
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
    print 'here'
    fractions = np.array([getFraction(x) for x in lp.tableau[rowInd, :]])
    print '1'
    pi, pi0 = fractions[:lp.nVariables], getFraction(lp.rhs[rowInd])
    print '2'
    pi_slacks = fractions[lp.nVariables:]
    print '3'
    pi -= pi_slacks * lp.coefMatrix
    print '4'
    pi0 -= np.dot(pi_slacks, lp.constraintsUpper)
    print 'done'
    if (abs(pi) > 1e-6).any():
        return pi, pi0
    return None, None

def getFraction(x):
    'Return the fraction part of x: x - floor(x)'
    return x - math.floor(x)

class GomoryCutGenerator:
    def __init__(self, cyLPModel):
        self.cyLPModel = cyLPModel

    def generateCuts(self, si, cglTreeInfo):
        m = self.cyLPModel
        x = m.getVarByName('x')

        clpModel = si.clpModel
        clpModel.dual(startFinishOptions='x')
        print clpModel.coefMatrix
        return []
        solution = clpModel.primalVariableSolution
        bv = clpModel.basicVariables
        rhs = clpModel.rhs

        intInds = clpModel.integerInformation

        rhsIsInt = map(isInt, rhs)

        cuts = []
        for rowInd in xrange(s.nConstraints):
            basicVarInd = bv[rowInd]
            if basicVarInd < clpModel.nVariables and intInds[basicVarInd] and not rhsIsInt[rowInd]:
                coef, b = gomoryCut(clpModel, rowInd)
                #print 'Adding cut: ', coef, '>=', b
                if b != None:
                    cuts.append(CyLPArray(coef) * x >= b)
                break

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
        cylpDir = os.environ['CYLP_SOURCE_DIR']
        #inputFile = os.path.join(cylpDir, 'CyLP', 'input', 'netlib', 'adlittle.mps')
        inputFile = os.path.join(cylpDir, 'CyLP', 'input', 'p0033.mps')
        m = s.extractCyLPModel(inputFile)
        #m = s.extractCyLPModel('/Users/mehdi/Downloads/timtab1.mps')
        x = m.getVarByName('x')
        s.setInteger(x)
        #s.setInteger(x[14])


    cbcModel = s.getCbcModel()

    gc = GomoryCutGenerator(m)
    cbcModel.addPythonCutGenerator(gc, name='PyGomory')

    cbcModel.branchAndBound()

    print cbcModel.primalVariableSolution

