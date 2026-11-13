'''
A cut generator written in Python, as an example for
:py:meth:`cylp.cy.CyCbcModel.CyCbcModel.addPythonCutGenerator`. It generates
Gomory's fractional cuts, which are valid for integer programs whose variables
are all integer and whose constraint coefficients and bounds are integers.

Cbc calls ``generateCuts`` on the problem after its preprocessing, while the
cuts are in terms of the model's variables. So in Cbc, this works only when
preprocessing leaves the columns unchanged, which it doesn't for p0033. See
`issue 255 <https://github.com/coin-or/CyLP/issues/255>`__.

Run as a script, it solves a small integer program with these cuts alone,
without branching.
'''
import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling import CyLPArray, CyLPModel

epsilon = 0.01

# Values this close to an integer are taken as integers
tolerance = 1e-9

# The size above which CyLP takes a bound as infinite
infinity = 1e16


def isInt(x):
    '''
    Return True if x, a number or a numpy array, is within ``epsilon`` of an
    integer, or all its elements are, and False otherwise
    '''
    return bool((np.abs(np.round(x) - x) < epsilon).all())


def getFraction(x):
    '''
    Return the fractional part of x, x - floor(x), or 0 if x is within
    ``tolerance`` of an integer. x is a number or a numpy array.
    '''
    f = x - np.floor(x)
    return np.where((f < tolerance) | (f > 1 - tolerance), 0., f)


def isIntegerProgram(lp):
    '''
    Return True if all of lp's variables are integer, and its constraint
    coefficients and finite bounds are integers
    '''
    bounds = np.concatenate([lp.variablesLower, lp.variablesUpper,
                             lp.constraintsLower, lp.constraintsUpper])
    return bool(np.all(lp.integerInformation) and
                isInt(lp.coefMatrix.data) and
                isInt(bounds[np.abs(bounds) < infinity]))


def getVariables(lp):
    '''
    Return the lower bounds, upper bounds and values of lp's variables
    followed by its slacks, and each constraint's ``bound``. As in
    :py:attr:`CyClpSimplex.rhs`, the slack of a constraint is ``bound`` minus
    the constraint's activity, where ``bound`` is the constraint's upper bound
    if it's finite, else its lower bound if that's finite, else 0.
    '''
    n = lp.nVariables
    rowLower = np.where(lp.constraintsLower > -infinity,
                        lp.constraintsLower, -np.inf)
    rowUpper = np.where(lp.constraintsUpper < infinity,
                        lp.constraintsUpper, np.inf)
    bound = np.where(rowUpper < np.inf, rowUpper,
                     np.where(rowLower > -np.inf, rowLower, 0.))
    colLower = np.where(lp.variablesLower > -infinity,
                        lp.variablesLower, -np.inf)
    colUpper = np.where(lp.variablesUpper < infinity,
                        lp.variablesUpper, np.inf)
    solution = lp.primalVariableSolutionAll
    lower = np.concatenate([colLower, bound - rowUpper])
    upper = np.concatenate([colUpper, bound - rowLower])
    value = np.concatenate([solution[:n], bound - solution[n:]])
    return lower, upper, value, bound


def gomoryCuts(lp):
    '''
    Return the Gomory cuts of lp, a CyClpSimplex object solved to optimality
    with ``startFinishOptions='x'``, as a list of pairs (pi, pi0), each the
    cut pi x >= pi0 in lp's variables x. There is a cut for each row of the
    tableau whose basic variable isn't a slack and has a fractional value.
    '''
    n = lp.nVariables
    lower, upper, value, bound = getVariables(lp)
    basicVariables = lp.basicVariables
    basic = np.zeros(len(value), bool)
    basic[basicVariables] = True

    # Gomory's cut needs each nonbasic variable to be 0 at the optimum. So
    # write each one as its distance y >= 0 from the bound it's at:
    # z = shift + sign * y, where z is the variable.
    atUpper = ~basic & (upper < np.inf) & (np.abs(value - upper) < 1e-7)
    atLower = ~basic & ~atUpper & (lower > -np.inf) & \
              (np.abs(value - lower) < 1e-7)
    notAtBound = ~basic & ~atUpper & ~atLower
    sign = np.where(atUpper, -1., 1.)
    shift = np.where(atUpper, upper, np.where(atLower, lower, 0.))

    tableau = lp.tableau
    rhs = lp.rhs
    A = lp.coefMatrix
    cuts = []
    for row, basicVar in enumerate(basicVariables):
        if basicVar >= n or isInt(rhs[row]):
            continue
        # This row is z_basicVar + sum_j tableau[row, j] z_j = constant. In
        # terms of the y's, which are 0 at the optimum, it's
        # z_basicVar + sum_j sign_j tableau[row, j] y_j = rhs[row]
        coefs = np.where(basic, 0., sign * tableau[row])
        if (notAtBound & (np.abs(coefs) > tolerance)).any():
            continue
        # Gomory's cut: sum_j f(coefs_j) y_j >= f(rhs[row]), where f is the
        # fractional part. Since y_j = sign_j (z_j - shift_j), it's
        # sum_j g_j z_j >= g0
        f = getFraction(coefs)
        g = f * sign
        g0 = getFraction(rhs[row]) + g[f != 0] @ shift[f != 0]
        # Substitute bound - A x for the slacks
        slackCoefs = g[n:]
        pi = g[:n] - np.asarray(A.T @ slackCoefs).ravel()
        pi0 = g0 - slackCoefs @ bound
        if (np.abs(pi) > 1e-6).any():
            cuts.append((pi, pi0))
    return cuts


class GomoryCutGenerator:
    '''
    Generate Gomory's fractional cuts in terms of ``cyLPModel``'s only
    variable, ``x``.
    '''
    def __init__(self, cyLPModel):
        self.cyLPModel = cyLPModel

    def cuts(self, lp):
        '''
        Return the Gomory cuts of lp, a CyClpSimplex object solved to
        optimality with ``startFinishOptions='x'``, as constraints that can
        be added to lp or returned to Cbc
        '''
        if not isIntegerProgram(lp):
            raise ValueError("GomoryCutGenerator needs an integer program "
                             "whose variables are all integer and whose "
                             "constraint coefficients and bounds are "
                             "integers")
        x = self.cyLPModel.getVarByName('x')
        return [CyLPArray(pi) * x >= pi0 for pi, pi0 in gomoryCuts(lp)]

    def generateCuts(self, si, cglTreeInfo):
        '''
        Called by Cbc. Return the Gomory cuts of the LP that Cbc is solving.
        '''
        clpModel = si.clpModel
        # tableau and rhs need the factorization that 'x' keeps
        if clpModel.dual(startFinishOptions='x') != 'optimal':
            return []
        return self.cuts(clpModel)


if __name__ == '__main__':
    m = CyLPModel()
    x = m.addVariable('x', 2, isInt=True)

    A = np.array([[7., -2.], [0., 1], [2., -2]])
    b = CyLPArray([14, 3, 3])
    m += A @ x <= b
    m += x >= 0
    m.objective = CyLPArray([-4, 1]) * x

    s = CyClpSimplex(m)
    s.logLevel = 0
    gc = GomoryCutGenerator(m)
    for _ in range(20):
        s.dual(startFinishOptions='x')
        cuts = gc.cuts(s)
        print('Objective', s.objectiveValue, 'at x =',
              s.primalVariableSolution['x'])
        if not cuts:
            break
        for cut in cuts:
            s += cut
