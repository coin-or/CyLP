import unittest
import types
import numpy as np

from cylp.cy import CyClpSimplex
from cylp.cy.CyClpSimplex import getMpsExample
from cylp.py.modeling.CyLPModel import CyLPArray
from cylp.py.pivots import DantzigPivot

# The properties that need Clp's factorization of the basis
factorizationProperties = ('basicVariables', 'rhs', 'basisInverse',
                           'tableau')


def getLP():
    s = CyClpSimplex()
    x = s.addVariable('x', 3)
    s += (np.array([[1., 2., 1.], [3., 1., 0.], [0., 1., 4.]]) @ x <=
          CyLPArray([4., 5., 6.]))
    s += x[0] + x[1] >= 0.5
    s += x >= 0
    s.objective = CyLPArray([-1., -1., -2.]) * x
    return s, x


def getScaledLP():
    # Clp scales p0033
    s = CyClpSimplex()
    s.readMps(getMpsExample())
    s.logLevel = 0
    return s


def basicValues(s, bv):
    '''
    The solution's values of the basic variables ``bv``. A slack's value is
    its constraint's upper bound, or its lower bound if the upper bound is
    infinite, minus its activity.
    '''
    n = s.nVariables
    solution = s.primalVariableSolutionAll
    lower, upper = s.constraintsLower, s.constraintsUpper
    bound = np.where(upper < 1e16, upper, np.where(lower > -1e16, lower, 0.))
    slacks = bound - solution[n:]
    return np.array([solution[j] if j < n else slacks[j - n] for j in bv])


class TestTableau(unittest.TestCase):

    def test_tableau(self):
        for getModel in (lambda: getLP()[0], getScaledLP):
            for solve in ('primal', 'dual'):
                s = getModel()
                self.assertEqual(getattr(s, solve)(startFinishOptions='x'),
                                 'optimal')
                m = s.nConstraints
                A = np.hstack([s.coefMatrix.toarray(), np.eye(m)])
                bv = s.basicVariables
                Binv = s.basisInverse
                self.assertTrue(np.allclose(Binv @ A[:, bv], np.eye(m)))
                self.assertTrue(np.allclose(s.tableau, Binv @ A))
                # On the scaled model, rhs mixed scaled and unscaled values
                self.assertTrue(np.allclose(s.rhs, basicValues(s, bv)))

    def test_rhsInPivotRule(self):
        # Pivot rules call getRightHandSide during the solve. The last call
        # is at the optimal basis, where no variable can enter.
        class RecordingPivot(DantzigPivot):
            def pivotColumn(self, *args):
                self.last = self.clpModel.basicVariables, self.clpModel.rhs
                return DantzigPivot.pivotColumn(self, *args)

        s = getScaledLP()
        pivot = RecordingPivot(s)
        s.setPivotMethod(pivot)
        self.assertEqual(s.primal(), 'optimal')
        bv, rhs = pivot.last
        self.assertTrue(np.allclose(rhs, basicValues(s, bv)))

    def test_noFactorization(self):
        # Clp aborted or segfaulted when there was no factorization, as when
        # IPython's tab completion read these properties
        s, x = getLP()

        def assertRaises():
            for name in factorizationProperties:
                self.assertRaises(RuntimeError, getattr, s, name)
            v = np.zeros(s.nConstraints)
            self.assertRaises(RuntimeError, s.getBInvCol, 0, v)
            self.assertRaises(RuntimeError, s.getBInvACol, 0, v)
            self.assertRaises(RuntimeError, s.getRightHandSide, v)

        assertRaises()
        s.primal()
        assertRaises()

        # After a solve that keeps the factorization, until a constraint is
        # added
        s.primal(startFinishOptions='x')
        self.assertEqual(s.tableau.shape, (4, 7))
        s += x[0] - x[1] <= 0.25
        assertRaises()
        s.primal(startFinishOptions='x')
        self.assertEqual(s.tableau.shape, (5, 8))

    def test_readEveryProperty(self):
        # As IPython's tab completion does
        names = [name for name in dir(CyClpSimplex) if isinstance(
                 getattr(CyClpSimplex, name), types.GetSetDescriptorType)]
        self.assertTrue(set(factorizationProperties) <= set(names))
        solved = getLP()[0]
        solved.primal()
        for s in (CyClpSimplex(), getLP()[0], solved):
            for name in names:
                try:
                    getattr(s, name)
                except Exception:
                    pass


if __name__ == '__main__':
    unittest.main()
