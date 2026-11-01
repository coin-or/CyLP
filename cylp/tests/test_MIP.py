import unittest
import inspect
import os
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex

from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
from cylp.py.mip import SimpleNodeCompare
from cylp.cy.CyCgl import CyCglGomory, CyCglClique, CyCglKnapsackCover

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestMIP(unittest.TestCase):

    def test_isInt(self):
        self.model = CyLPModel()
        model = self.model

        x = model.addVariable('x', 3, isInt=True)
        y = model.addVariable('y', 2)

        A = np.array([[1., 2., 0],[1., 0, 1.]])
        B = np.array([[1., 0, 0], [0, 0, 1.]])
        D = np.array([[1., 2.],[0, 1]])
        a = CyLPArray([5, 2.5])
        b = CyLPArray([4.2, 3])
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A @ x <= a)
        model.addConstraint(2 <= B @ x + D @ y <= b)
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y.sum()


        self.s = CyClpSimplex(model)
        s = self.s

        cbcModel = s.getCbcModel()

        cbcModel.solve()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue((abs(sol_x -
                        np.array([0, 2, 2]) ) <= 10**-6).all())
        sol_y = cbcModel.primalVariableSolution['y']
        self.assertTrue((abs(sol_y -
                        np.array([0, 1]) ) <= 10**-6).all())

    def test_SetInt_CopyIn(self):
        self.model = CyLPModel()
        model = self.model

        x = model.addVariable('x', 3)
        y = model.addVariable('y', 2)

        A = np.array([[1., 2., 0],[1., 0, 1.]])
        B = np.array([[1., 0, 0], [0, 0, 1.]])
        D = np.array([[1., 2.],[0, 1]])
        a = CyLPArray([5, 2.5])
        b = CyLPArray([4.2, 3])
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A @ x <= a)
        model.addConstraint(2 <= B @ x + D @ y <= b)
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y.sum()


        self.s = CyClpSimplex(model)
        s = self.s
        s.setInteger(x[1:3])

        cbcModel = s.getCbcModel()
        cbcModel.solve()

        # Every x = [t, 2, 2], y = [0, 1 - t/2] with 0 <= t <= 0.5 is optimal,
        # so only check the objective and the integer variables
        self.assertTrue(abs(cbcModel.objectiveValue - 4.0) <= 10**-6)
        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue((abs(sol_x[1:3] -
                        np.array([2, 2]) ) <= 10**-6).all())


        s.copyInIntegerInformation(np.array(
                            [True, False, False, False, True], np.uint8))
        cbcModel = s.getCbcModel()
        cbcModel.solve()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue((abs(sol_x -
                        np.array([0, 2, 1.1]) ) <= 10**-6).all())
        sol_y = cbcModel.primalVariableSolution['y']
        self.assertTrue((abs(sol_y -
                        np.array([0, 1]) ) <= 10**-6).all())

    def test_NodeCompare(self):
        s = CyClpSimplex()
        self.assertEqual(s.readMps(join(currentFilePath, '../input/p0033.mps')), 0)

        s.copyInIntegerInformation(np.array(s.nCols * [True], np.uint8))

        print("Solving relaxation")
        cbcModel = s.getCbcModel()
        n = SimpleNodeCompare()
        cbcModel.setNodeCompare(n)

        gom = CyCglGomory(limit=150)
        #gom.limit = 150
        cbcModel.addCutGenerator(gom, name="Gomory")

        #clq = CyCglClique()
        #cbcModel.addCutGenerator(clq, name="Clique")

        knap = CyCglKnapsackCover(maxInKnapsack=50)
        cbcModel.addCutGenerator(knap, name="Knapsack")


        cbcModel.solve()
        self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)

    def test_NodeCompareCallbacks(self):
        # p0033 is solved at the root, so the node comparison is never
        # called. This knapsack problem needs a tree search.
        rs = np.random.RandomState(0)
        A = rs.randint(10, 100, size=(5, 50)).astype(float)
        c = rs.randint(10, 100, size=50).astype(float)

        s = CyClpSimplex()
        x = s.addVariable('x', 50, isInt=True)
        s += A @ x <= CyLPArray(A.sum(axis=1) / 2)
        s += 0 <= x <= 1
        s.objective = -CyLPArray(c) * x

        cbcModel = s.getCbcModel()
        n = SimpleNodeCompare()
        cbcModel.setNodeCompare(n)
        cbcModel.solve()

        # SimpleNodeCompare.newSolution switches to breadth-first
        self.assertEqual(n.method, 'breadth')
        self.assertTrue(abs(cbcModel.objectiveValue + 1837.0) < 10 ** -6)

    def test_multiDim(self):
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray
        s = CyClpSimplex()
        x = s.addVariable('x', (5, 3, 6))
        s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5.5
        s += 0 <= x <= 2.2
        c = CyLPArray(range(18))
        s.objective = c * x[2, :, :] + c * x[0, :, :]

        s.setInteger(x)

        cbcModel = s.getCbcModel()
        cbcModel.solve()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue(abs(sol_x[0, 1, 0] - 1) <= 10**-6)
        self.assertTrue(abs(sol_x[2, 0, 3] - 2) <= 10**-6)

    def test_multiDim_Cbc_solve(self):
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray
        s = CyClpSimplex()
        x = s.addVariable('x', (5, 3, 6))
        s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5.5
        s += 0 <= x <= 2.2
        c = CyLPArray(range(18))
        s.objective = c * x[2, :, :] + c * x[0, :, :]

        s.setInteger(x)

        cbcModel = s.getCbcModel()
        cbcModel.solve()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue(abs(sol_x[0, 1, 0] - 1) <= 10**-6)
        self.assertTrue(abs(sol_x[2, 0, 3] - 2) <= 10**-6)

    def test_constraintSolutions(self):
        # Two suppliers with minimum amounts (#130). Supplier 1 meets the
        # demand, so its cost is the marginal price of the demand
        model = CyLPModel()
        s1 = model.addVariable('supply_1', 1)
        s2 = model.addVariable('supply_2', 1)
        w1 = model.addVariable('switch_1', 1, isInt=True)
        w2 = model.addVariable('switch_2', 1, isInt=True)
        model += 0 <= s1 <= 100
        model += 0 <= s2 <= 100
        model += 0 <= w1 <= 1
        model += 0 <= w2 <= 1
        model += s1 - 50 * w1 >= 0
        model += s1 - 100 * w1 <= 0
        model += s2 - 70 * w2 >= 0
        model += s2 - 100 * w2 <= 0
        model.addConstraint(s1 + s2 == 60, 'demand')
        model.objective = 10 * s1 + 8 * s2

        cbcModel = CyClpSimplex(model).getCbcModel()
        cbcModel.solve()

        self.assertTrue((abs(cbcModel.primalColumnSolution -
                             np.array([60, 0, 1, 0])) <= 10**-6).all())
        self.assertTrue(abs(cbcModel.primalConstraintSolution['demand'] - 60)
                        <= 10**-6)
        self.assertTrue(abs(cbcModel.dualConstraintSolution['demand'] - 10)
                        <= 10**-6)

        # The arrays are copies, so changing them doesn't change the model
        sol = cbcModel.primalColumnSolution
        sol[:] = -1
        self.assertTrue(abs(cbcModel.primalColumnSolution[0] - 60) <= 10**-6)

    def test_dualVariableSolution(self):
        s = CyClpSimplex()
        x = s.addVariable('x', 2, isInt=True)
        y = s.addVariable('y', (2, 2))
        s += 0 <= x <= 3
        s += 0 <= y <= 1
        s += x.sum() + y[0, :].sum() >= 1.5
        s.objective = x.sum() + CyLPArray([4., 5., 6., 7.]) @ y[:, :]

        cbcModel = s.getCbcModel()
        cbcModel.solve()

        # y is 0 and its constraint isn't binding, so its reduced costs are
        # its costs
        dj = cbcModel.dualVariableSolution
        self.assertTrue(abs(dj['y'][1, 0] - 6) <= 10**-6)
        self.assertTrue(abs(dj['y'][1, 1] - 7) <= 10**-6)
        self.assertTrue((abs(cbcModel.dualColumnSolution[2:] -
                             np.array([4, 5, 6, 7])) <= 10**-6).all())

if __name__ == '__main__':
    unittest.main()

