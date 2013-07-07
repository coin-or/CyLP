import unittest
import inspect
import os
from os.path import join
import numpy as np

from CyLP.cy import CyClpSimplex

from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray
from CyLP.py.utils.sparseUtil import csr_matrixPlus, csc_matrixPlus

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestModel(unittest.TestCase):

    def setUp(self):
        'Test remove constraint'

        model = CyLPModel()

        x = model.addVariable('x', 3)
        y = model.addVariable('y', 2)

        A = csc_matrixPlus(([1, 2, 1, 1], ([0, 0, 1, 1], [0, 1, 0, 2])),  shape=(2, 3))
        B = csc_matrixPlus(([1, 1], ([0, 1], [0, 2])),  shape=(2, 3))
        D = np.matrix([[1., 2.],[0, 1]])
        a = CyLPArray([3, 2.5])
        b = CyLPArray([4.2, 3])
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A * x <= a, 'res1')
        model.addConstraint(2 <= B * x + D * y <= b, 'res2')
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)
        model.addConstraint(x[0] >= 0.1)

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y[0] + 2 * y[1]

        s = CyClpSimplex(model)
        self.s = s
        self.x = x
        self.y = y

    def test_Variable(self):
        s = self.s
        x = self.x
        s.setVariableStatus(x[1:3], 'atUpperBound')
        self.assertTrue((s.getVariableStatus(x) ==
                    ['atLowerBound', 'atUpperBound', 'atUpperBound']).all())

    def test_Variable2(self):
        s = self.s
        x = self.x
        s.setVariableStatus(x, 'atUpperBound')
        self.assertTrue((s.getVariableStatus(x) ==
                    3 * ['atUpperBound']).all())

    def test_Variable3(self):
        s = self.s
        x = self.x
        s.setVariableStatus(x[1], 'atUpperBound')
        self.assertTrue(s.getVariableStatus(x[1]) == 'atUpperBound')

    def test_Constraint(self):
        s = self.s
        s.setConstraintStatus('res2', 'atLowerBound')
        self.assertTrue((s.getConstraintStatus('res2') ==
                    2 * ['atLowerBound']).all())

    def test_Constraint2(self):
        s = self.s
        s.setConstraintStatus('res1', 'atUpperBound')
        self.assertTrue((s.getConstraintStatus('res1') ==
                    2 * ['atUpperBound']).all())

if __name__ == '__main__':
    unittest.main()

