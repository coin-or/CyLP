import unittest
import inspect
import os
from os.path import join
import numpy as np

from CyLP.cy import CyClpSimplex

from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray
from CyLP.py.mip import SimpleNodeCompare
from CyLP.cy.CyCgl import CyCglGomory, CyCglClique, CyCglKnapsackCover

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
        model.objective = c * x + 2 * y


        self.s = CyClpSimplex(model)
        s = self.s

        cbcModel = s.getCbcModel()

        cbcModel.branchAndBound()

        sol = cbcModel.primalVariableSolution

        self.assertTrue((abs(sol -
                        np.array([0, 2, 2, 0, 1]) ) <= 10**-6).all())

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
        model.objective = c * x + 2 * y


        self.s = CyClpSimplex(model)
        s = self.s
        s.setInteger(x[1:3])

        cbcModel = s.getCbcModel()
        cbcModel.branchAndBound()

        sol = cbcModel.primalVariableSolution
        self.assertTrue((abs(sol -
                        np.array([0.5, 2, 2, 0, 0.75]) ) <= 10**-6).all())


        s.copyInIntegerInformation(np.array(
                            [True, False, False, False, True], np.uint8))
        cbcModel = s.getCbcModel()
        cbcModel.branchAndBound()

        sol = cbcModel.primalVariableSolution
        self.assertTrue((abs(sol -
                        np.array([0, 2, 1.1, 0, 1]) ) <= 10**-6).all())

    def test_NodeCompare(self):
        s = CyClpSimplex()
        s.readMps(join(currentFilePath, '../input/p0033.mps'))

        s.copyInIntegerInformation(np.array(s.nCols * [True], np.uint8))

        print "Solving relaxation"
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


if __name__ == '__main__':
    unittest.main()

