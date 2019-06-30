from __future__ import print_function
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

        cbcModel = s.getCbcModel()

        cbcModel.branchAndBound()

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
        s.setInteger(x[1:3])

        cbcModel = s.getCbcModel()
        cbcModel.branchAndBound()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue((abs(sol_x -
                        np.array([0.5, 2, 2]) ) <= 10**-6).all())
        sol_y = cbcModel.primalVariableSolution['y']
        self.assertTrue((abs(sol_y -
                        np.array([0, 0.75]) ) <= 10**-6).all())


        s.copyInIntegerInformation(np.array(
                            [True, False, False, False, True], np.uint8))
        cbcModel = s.getCbcModel()
        cbcModel.branchAndBound()

        sol_x = cbcModel.primalVariableSolution['x']
        self.assertTrue((abs(sol_x -
                        np.array([0, 2, 1.1]) ) <= 10**-6).all())
        sol_y = cbcModel.primalVariableSolution['y']
        self.assertTrue((abs(sol_y -
                        np.array([0, 1]) ) <= 10**-6).all())

    def test_NodeCompare(self):
        s = CyClpSimplex()
        s.readMps(join(currentFilePath, '../input/p0033.mps'))

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


        cbcModel.branchAndBound()
        self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)

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
        cbcModel.branchAndBound()

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

if __name__ == '__main__':
    unittest.main()

