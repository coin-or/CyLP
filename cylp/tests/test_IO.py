import unittest
import os
import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray


class TestMIP(unittest.TestCase):

    def test_isInt(self):
        self.model = CyLPModel()
        model = self.model

        x = model.addVariable('x', 3, isInt=True)
        y = model.addVariable('y', 2)

        A = np.matrix([[1., 2., 0],[1., 0, 1.]])
        B = np.matrix([[1., 0, 0], [0, 0, 1.]])
        D = np.matrix([[1., 2.],[0, 1]])
        a = CyLPArray([5, 2.5])
        b = CyLPArray([4.2, 3])
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A*x <= a)
        model.addConstraint(2 <= B * x + D * y <= b)
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y.sum()


        self.s = CyClpSimplex(model)
        s = self.s

        s.writeMps("cylptest.mps")
        os.remove("cylptest.mps")
        s.writeMps("cylptest.lp")
        os.remove("cylptest.lp")



if __name__ == '__main__':
    unittest.main()

