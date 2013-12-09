import numpy as np
#from scipy import sparse
from cylp.cy import CyCoinMpsIO
from cylp.py.utils.sparseUtil import csr_matrixPlus


def readQPS(inputFilename):
    problem = CyCoinMpsIO()
    problem.readMps(inputFilename)

    n = problem.nVariables
    m = problem.nConstraints

    signs = problem.constraintSigns
    iEq = [i for i in range(len(signs)) if chr(signs[i]) == 'E']
    numberOfEqualities = len(iEq)
    iInEq = [i for i in range(len(signs)) if i not in iEq]
    numberOfInequalities = len(iInEq)

    c = problem.matrixByRow
    el = c.elements
    col = c.indices
    start = c.vectorStarts
    coefs = csr_matrixPlus((el, col, start), shape=(m, n))

    # an invalid value for initialization
    # This wont make any problems because we always
    # check number of equalties and
    # inequalities first, before accessing them
    A = C = b = c_up = c_low = 0

    rhs = problem.rightHandSide

    if numberOfEqualities:
        A = csr_matrixPlus(coefs[iEq])
        b = rhs[iEq]

    if numberOfInequalities:
        C = csr_matrixPlus(coefs[iInEq])
        c_up = problem.constraintUpper[iInEq]
        c_low = problem.constraintLower[iInEq]

    Hessian = problem.Hessian

    x_low = problem.variableLower
    x_up = problem.variableUpper

    c = problem.objCoefficients

    return (Hessian, c, A, b, C, c_low, c_up, x_low, x_up,
            n, len(iEq), len(iInEq), problem.objectiveOffset)
