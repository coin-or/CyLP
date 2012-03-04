import numpy as np
#from scipy import sparse
from CyCoinMpsIO import CyCoinMpsIO
from CyLP.py.utils.sparseUtil import csr_matrixPlus


def readQPS(inputFilename):
    problem = CyCoinMpsIO()
    problem.readMps(inputFilename)

    n = problem.getNumCols()
    m = problem.getNumRows()

    signs = problem.getRowSense()
    iEq = [i for i in range(len(signs)) if chr(signs[i]) == 'E']
    numberOfEqualities = len(iEq)
    iInEq = [i for i in range(len(signs)) if i not in iEq]
    numberOfInequalities = len(iInEq)

    c = problem.getMatrixByRow()
    el = c.getElements()
    col = c.getIndices()
    start = c.getVectorStarts()
    coefs = csr_matrixPlus((el, col, start), shape=(m, n))

    # an invalid value for initialization
    # This wont make any problems because we always
    # check number of equalties and
    # inequalities first, before accessing them
    A = C = b = c_up = c_low = 0

    rhs = problem.getRightHandSide()

    if numberOfEqualities:
        A = csr_matrixPlus(coefs[iEq])
        b = rhs[iEq]

    if numberOfInequalities:
        C = csr_matrixPlus(coefs[iInEq])
        c_up = problem.getRowUpper()[iInEq]
        c_low = problem.getRowLower()[iInEq]

    Hessian = problem.getHessian()

    x_low = problem.getColLower()
    x_up = problem.getColUpper()

    c = problem.getObjCoefficients()

    return (Hessian, c, A, b, C, c_low, c_up, x_low, x_up,
            n, len(iEq), len(iInEq), problem.getObjectiveOffset())
