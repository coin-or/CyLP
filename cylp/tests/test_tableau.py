import unittest
import types
import numpy as np

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

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


class TestTableau(unittest.TestCase):

    def test_tableau(self):
        for solve in ('primal', 'dual'):
            s, x = getLP()
            self.assertEqual(getattr(s, solve)(startFinishOptions='x'),
                             'optimal')
            n, m = s.nVariables, s.nConstraints
            A = np.hstack([s.coefMatrix.toarray(), np.eye(m)])
            bv = s.basicVariables
            Binv = s.basisInverse
            self.assertTrue(np.allclose(Binv @ A[:, bv], np.eye(m)))
            self.assertTrue(np.allclose(s.tableau, Binv @ A))
            # The basic variables' values
            sol = s.primalVariableSolution['x']
            structural = bv < n
            self.assertTrue(np.allclose(s.rhs[structural],
                                        sol[bv[structural]]))

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
