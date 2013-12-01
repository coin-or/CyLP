import inspect
import os
import unittest
import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.utils.sparseUtil import csr_matrixPlus
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray, getCoinInfinity

inf = getCoinInfinity()
currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestModeling(unittest.TestCase):

    def setUp(self):
        self.model = CyLPModel()
        model = self.model

        self.x = model.addVariable('x', 5)


        self.b = np.array([3.1, 4.2])

        self.A = np.matrix([[1, 2, 3, 3, 4], [3, 2, 1, 2, 5]])
        self.B = np.matrix([[0, 0, 0, -1.5, -1.5], [-3, 0, 0, -2,0 ]])
        self.D = np.matrix([[1, 3, 1, 4], [2, 1,3, 5]])

    def test_Obj1(self):
        m = self.model
        x = self.x
        y = m.addVariable('y', 4)
        z = m.addVariable('z', 5)
        A = self.A
        b = self.b

        k = m.addVariable('k', 2)


        m.addConstraint(x >= 0)

        m.addConstraint(A * x <= b)

        m.addConstraint(z >= 0)

        m.objective = x.sum() + z.sum() + k.sum()


    def test_bound1(self):
        model = self.model
        x = self.x

        model.addConstraint(x[2:4] >= 3)
        self.assertTrue((x.lower == np.array([-inf, -inf, 3, 3, -inf])).all())
        self.assertTrue((x.upper == np.array(5* [inf])).all())

    def test_bound2(self):
        model = self.model
        x = self.x

        model.addConstraint(x[2:4] <= 3)
        self.assertTrue((x.lower == np.array(5 * [-inf])).all())
        self.assertTrue((x.upper == np.array([inf, inf, 3, 3, inf])).all())

    def test_bound3(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(1.1 <= x[2:4] <= b)
        self.assertTrue((x.lower == np.array(
                            [-inf, -inf, 1.1, 1.1, -inf])).all())
        self.assertTrue((x.upper == np.array(
                            [inf, inf, b[0], b[1], inf])).all())
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((vu[:5] == np.array(
                                    [inf, inf, b[0], b[1], inf])).all())
        self.assertTrue((vl[:5] == np.array(
                                    [-inf, -inf, 1.1, 1.1, -inf])).all())

    def test_bound4(self):
        model = self.model
        x = self.x
        b = self.b

        model += x[2:4] == b
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((vu[:5] == np.array(
                                    [inf, inf, b[0], b[1], inf])).all())
        self.assertTrue((vl[:5] == np.array(
                                    [-inf, -inf, b[0], b[1], -inf])).all())

    def test_constraint_single1(self):
        model = self.model
        x = self.x

        model += 1.1 <= x[0] - 3 * x[1:3].sum() + 2 * x[2:5].sum() <= 4.5
        cons = model.constraints[0]
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((m.todense() == np.array(
                            [1, -3, -1, 2, 2])).all())

    def test_constraint_single2(self):
        model = self.model
        x = self.x

        model.addConstraint(-x[1] + 3 * x[1:4].sum() + None * x >= 4.5)
        cons = model.constraints[0]
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((m.todense() == np.array(
                            [0, 2, 3, 3, 0])).all())
        self.assertTrue((cons.lower == np.array(
                            [4.5])).all())
        self.assertTrue((cons.upper == np.array(
                            [inf])).all())


    def test_constraint_single3(self):
        model = self.model
        x = self.x


        k = csr_matrixPlus([[3.1], [4.2]])
        model.addConstraint(-x[0] + -3 * x[1] + k.T * x[1:3]  >= 4.5)
        cons = model.constraints[0]
        m, cl, cu, vl, vu  = model.makeMatrices()
        self.assertTrue((abs(m.todense()-np.array(
                            [-1, 0.1, 4.2, 0, 0])) < 0.000001).all())

    def test_constraint_1(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(2 <= -x <= 4.5)

        cons = model.constraints[0]

        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((abs(m.todense() +
                        np.eye(5, 5)) < 0.000001).all())

        #self.assertEqual(self.c[4], 3.3)

    def test_removeConst(self):

        m = self.model
        x = self.x
        A = self.A
        b = self.b


        m.addConstraint(x >= 0)

        m.addConstraint(A * x == b)

        m.addConstraint(x[1:3].sum() >= 1, 'rr')

        m.objective = x.sum()
        s = CyClpSimplex(m)
        s.primal()
        self.assertAlmostEqual(s.primalVariableSolution['x'][1], 1, 7)

        s.removeConstraint('rr')
        s.primal()
        self.assertAlmostEqual(s.primalVariableSolution['x'][1], 0, 7)

    def test_removeVar(self):
        m = self.model
        x = self.x
        A = self.A
        B = self.B
        D = self.D
        b = self.b

        y = m.addVariable('y', 4)
        z = m.addVariable('z', 5)

        m.addConstraint(x >= 0)
        m.addConstraint(y >= -10)
        m.addConstraint(z >= -10)

        m.addConstraint(A * x + D * y + B * z <= b)
        m += x[0] + y[0] + z[0] >= 1.12

        m.objective = x.sum() + y.sum() + z.sum()
        s = CyClpSimplex(m)
        s.primal()

        self.assertTrue('y' in s.primalVariableSolution.keys())

        m.removeVariable('y')
        s = CyClpSimplex(m)
        s.primal()
        self.assertTrue('y' not in s.primalVariableSolution.keys())

    def test_removeVar2(self):
        s = CyClpSimplex()
        fp = os.path.join(currentFilePath, '../../input/p0033.mps')
        s.extractCyLPModel(fp)
        y = s.addVariable('y', 3)
        s.primal()

        x = s.getVarByName('x')
        s.addConstraint(x[1] +  y[1] >= 1.2)
        #s.primal()
        s.removeVariable('x')
        s.primal()
        s = s.primalVariableSolution
        self.assertTrue((s['y'] - np.array([0, 1.2, 0]) <= 10**-6).all())


if __name__ == '__main__':
    unittest.main()

