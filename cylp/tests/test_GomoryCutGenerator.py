import itertools
import unittest
import numpy as np

from cylp.cy import CyClpSimplex
from cylp.cy.CyClpSimplex import getMpsExample
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
from cylp.py.mip.GomoryCutGenerator import GomoryCutGenerator, gomoryCuts


def getIP(A, b, c, upper=None):
    '''
    Return min c x subject to A x <= b and 0 <= x <= upper, x integer, as a
    CyClpSimplex object and its model
    '''
    m = CyLPModel()
    x = m.addVariable('x', len(c), isInt=True)
    m += np.array(A, float) @ x <= CyLPArray(b)
    if upper is None:
        m += x >= 0
    else:
        m += 0 <= x <= CyLPArray(upper)
    m.objective = CyLPArray(c) * x
    s = CyClpSimplex(m)
    s.logLevel = 0
    return s, m


def integerPoints(A, b, upper):
    '''
    The integer points x with A x <= b and 0 <= x <= upper, as the rows of
    an array
    '''
    points = np.array(list(itertools.product(
        *[range(int(u) + 1) for u in upper])), float)
    return points[(points @ np.array(A, float).T <= np.array(b) + 1e-9).all(
        axis=1)]


class TestGomoryCutGenerator(unittest.TestCase):

    def checkCuts(self, A, b, c, upper, boundsInModel=True):
        '''
        Check that each Gomory cut of the integer program cuts off the LP's
        optimum and no integer point. Return the number of cuts.
        '''
        s, m = getIP(A, b, c, upper if boundsInModel else None)
        self.assertEqual(s.dual(startFinishOptions='x'), 'optimal')
        x = s.primalVariableSolution['x']
        points = integerPoints(A, b, upper)
        cuts = gomoryCuts(s)
        for pi, pi0 in cuts:
            self.assertLess(pi @ x, pi0 - 1e-7)
            self.assertTrue((points @ pi >= pi0 - 1e-7).all())
        return len(cuts)

    def test_variableAtUpperBound(self):
        # x[1] is nonbasic at its upper bound. The cut was x[0] <= 2/3, which
        # cuts off (1, 0).
        self.assertGreater(self.checkCuts([[3, 2], [4, 2], [7, 5]], [3, 5, 12],
                                          [-4, -4], [3, 1]), 0)

    def test_valueBelowAnInteger(self):
        # x[1] is 6.999999999999999 in rhs. It was taken as fractional, and
        # the cut 3 x[0] + x[1] <= 6 cut off the optimum, (0, 7).
        self.checkCuts([[6, 2], [0, 0], [6, 2]], [18, 5, 14], [-2, -3],
                       [10, 10], boundsInModel=False)

    def test_randomIntegerPrograms(self):
        rng = np.random.default_rng(1)
        numberOfCuts = 0
        for boundsInModel in (True, False):
            for _ in range(100):
                n, k = rng.integers(2, 4), rng.integers(2, 4)
                if boundsInModel:
                    A = rng.integers(-3, 8, size=(k, n))
                    upper = rng.integers(1, 4, size=n)
                else:
                    # Keep the feasible region bounded
                    A = rng.integers(0, 8, size=(k, n))
                    upper = np.full(n, 19)
                A[0] = rng.integers(1, 8, size=n)
                b = rng.integers(3, 20, size=k)
                c = -rng.integers(1, 6, size=n)
                numberOfCuts += self.checkCuts(A, b, c, upper, boundsInModel)
        self.assertGreater(numberOfCuts, 100)

    def test_cuttingPlanes(self):
        # The example in GomoryCutGenerator's __main__
        s, m = getIP([[7, -2], [0, 1], [2, -2]], [14, 3, 3], [-4, 1])
        generator = GomoryCutGenerator(m)
        for _ in range(10):
            self.assertEqual(s.dual(startFinishOptions='x'), 'optimal')
            cuts = generator.cuts(s)
            if not cuts:
                break
            for cut in cuts:
                s += cut
        self.assertAlmostEqual(s.objectiveValue, -7)
        self.assertTrue(np.allclose(s.primalVariableSolution['x'], [2, 1]))

    def test_p0033(self):
        # p0033's variables are binary, so many are nonbasic at their upper
        # bound
        cbcModel = CyClpSimplex()
        cbcModel.logLevel = 0
        mm = cbcModel.extractCyLPModel(getMpsExample())
        cbcModel.setInteger(mm.getVarByName('x'))
        cbcModel = cbcModel.getCbcModel()
        cbcModel.logLevel = 0
        cbcModel.solve()
        optimum = cbcModel.primalVariableSolution['x']

        s = CyClpSimplex()
        s.logLevel = 0
        # With Clp's scaling, these LPs' solutions can be optimal for the
        # scaled problem but not for the unscaled one
        s.scaling = 0
        m = s.extractCyLPModel(getMpsExample())
        s.setInteger(m.getVarByName('x'))
        generator = GomoryCutGenerator(m)
        objective = -np.inf
        for _ in range(5):
            self.assertEqual(s.primal(startFinishOptions='x'), 'optimal')
            self.assertGreaterEqual(s.objectiveValue, objective - 1e-7)
            objective = s.objectiveValue
            x = s.primalVariableSolution['x']
            cuts = gomoryCuts(s)
            self.assertGreater(len(cuts), 0)
            for pi, pi0 in cuts:
                self.assertLess(pi @ x, pi0 - 1e-7)
                self.assertGreaterEqual(pi @ optimum, pi0 - 1e-7)
            for cut in generator.cuts(s):
                s += cut
        self.assertGreater(objective, 2520.6)

    def test_notAnIntegerProgram(self):
        m = CyLPModel()
        x = m.addVariable('x', 2)
        m += np.array([[1., 1.]]) @ x <= 1.5
        m += x >= 0
        m.objective = CyLPArray([-1., -1.]) * x
        s = CyClpSimplex(m)
        s.logLevel = 0
        s.dual(startFinishOptions='x')
        self.assertRaises(ValueError, GomoryCutGenerator(m).cuts, s)


if __name__ == '__main__':
    unittest.main()
