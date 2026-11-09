import unittest
import inspect
import os
from os.path import join
import numpy as np
from scipy import sparse

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from cylp.py.pivots import WolfePivot

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestQP(unittest.TestCase):
    def test_Hessian(self):
        'Set the quadratic part of the objective with a sparse matrix'
        # x0^2 + x0 x1 + x1^2 - 3 x0 - 3 x1 is smallest at (1, 1). Clp
        # counts the elements off the diagonal twice if it gets both.
        s = CyClpSimplex()
        x = s.addVariable('x', 2)
        s += x[0] + x[1] <= 10
        s += 0 <= x <= 10
        s.objective = CyLPArray([-3., -3.]) * x
        s.Hessian = sparse.csr_matrix([[2., 1.], [1., 2.]])
        self.assertEqual(s.primal(), 'optimal')
        self.assertTrue((abs(s.primalVariableSolution['x'] -
                             np.array([1, 1])) <= 10**-6).all())
        self.assertTrue(abs(s.objectiveValue + 3) <= 10**-6)

        # The rows and columns after the first are zero, and the
        # elements are integers
        s = CyClpSimplex()
        x = s.addVariable('x', 3)
        s += x.sum() <= 10
        s += 0 <= x <= 10
        s.objective = CyLPArray([-2., 1., 1.]) * x
        s.Hessian = sparse.coo_matrix(([2], ([0], [0])), shape=(3, 3))
        self.assertEqual(s.primal(), 'optimal')
        self.assertTrue((abs(s.primalVariableSolution['x'] -
                             np.array([1, 0, 0])) <= 10**-6).all())
        self.assertTrue(abs(s.objectiveValue + 1) <= 10**-6)

        # The Hessian read from a QPS file gives the same problem again
        qps = join(currentFilePath, '../input/hs35.qps')
        s = CyClpSimplex()
        self.assertEqual(s.readMps(qps), 0)
        H = s.Hessian
        s = CyClpSimplex()
        self.assertEqual(s.readMps(qps), 0)
        s.Hessian = H
        self.assertEqual(s.primal(), 'optimal')
        self.assertTrue(abs(s.objectiveValue - 1. / 9) <= 10**-6)

    def test_1(self):
        """simplest QP test"""
        s = CyClpSimplex()
        self.assertEqual(s.readMps(join(currentFilePath, '../input/hs35.qps')), 0)
        #self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)

        #print(s.Hessian.todense())

        p = WolfePivot(s)
        s.setPivotMethod(p)

        s.primal()
        print(s.primalVariableSolution)
        print(s.objectiveValue)

if __name__ == '__main__':
    unittest.main()

