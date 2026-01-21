import unittest
import inspect
import os
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex

from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
from cylp.py.utils.sparseUtil import csr_matrixPlus, csc_matrixPlus

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestModel(unittest.TestCase):

    def test(self):
        model = CyLPModel()

        x = model.addVariable('x', 3)

        A = np.matrix([[1,2,3], [1,1,1]])
        b = CyLPArray([5, 3])

        model.addConstraint(A * x == b)
        model.addConstraint(x >= 0)

        model.objective = 1*x[0]  + 1*x[1] + 1.1 * x[2]

        # Solve it a first time
        s = CyClpSimplex(model)
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([1,2,0]) ) <= 10**-6).all())
        # Add a cut
        s.addConstraint(x[0] >= 1.1)
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([1.1, 1.8, 0.1]) ) <= 10**-6).all())

        # Change the objective function
        c = csr_matrixPlus([[1, 10, 1.1]]).T
        s.objective = c.T * x
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([2, 0, 1]) ) <= 10**-6).all())

    def test2(self):
        'Same as test1, but use cylp indirectly.'
        s = CyClpSimplex()

        x = s.addVariable('x', 3)

        A = np.matrix([[1,2,3], [1,1,1]])
        b = CyLPArray([5, 3])

        s += A * x == b
        s += x >= 0

        s.objective = 1 * x[0] + 1 * x[1] + 1.1 * x[2]

        # Solve it a first time
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([1,2,0]) ) <= 10**-6).all())
        # Add a cut
        s.addConstraint(x[0] >= 1.1)
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([1.1, 1.8, 0.1]) ) <= 10**-6).all())

        # Change the objective function
        c = csr_matrixPlus([[1, 10, 1.1]]).T
        s.objective = c.T * x
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue((abs(sol - np.array([2, 0, 1]) ) <= 10**-6).all())


    def test_removeConstraint(self):
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
        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0.1, 1.45, 1.1, 0, 0.95]) ) <= 10**-6).all())

        s.removeConstraint('res2')
        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0.1, 1.45, 1.1, 0, 0]) ) <= 10**-6).all())

        s.removeConstraint('res1')
        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0.1, 2, 1.1, 0, 0]) ) <= 10**-6).all())

    def test_multiVar(self):
        model = CyLPModel()

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
        model.objective = c * x + 2 * y[0] + 2 * y[1]


        s = CyClpSimplex(model)

        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0.2, 2, 1.1, 0, 0.9]) ) <= 10**-6).all())


        s += x[2] + y[1] >= 2.1
        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0, 2, 1.1, 0, 1]) ) <= 10**-6).all())

    def test_onlyBounds(self):
        model = CyLPModel()

        x = model.addVariable('x', 3)
        y = model.addVariable('y', 2)

        model += y >= 1
        model += 2 <= x <= 4

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y[0] + 2 * y[1]

        s = CyClpSimplex(model)
        s.primal()

        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([2, 4, 2, 1, 1]) ) <= 10**-6).all())
    def test_onlyBounds2(self):
        s = CyClpSimplex()

        x = s.addVariable('x', 3)
        y = s.addVariable('y', 2)

        s += y >= 1
        s += 2 <= x <= 4

        c = CyLPArray([1., -2., 3.])
        s.objective = c * x + 2 * y[0] + 2 * y[1]

        s.primal()

        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([2, 4, 2, 1, 1]) ) <= 10**-6).all())

    def test_Sparse(self):
        model = CyLPModel()

        x = model.addVariable('x', 3)
        y = model.addVariable('y', 2)

        A = csc_matrixPlus(([1, 2, 1, 1], ([0, 0, 1, 1], [0, 1, 0, 2])),  shape=(2, 3))
        B = csc_matrixPlus(([1, 1], ([0, 1], [0, 2])),  shape=(2, 3))
        D = np.matrix([[1., 2.],[0, 1]])
        a = CyLPArray([5, 2.5])
        b = CyLPArray([4.2, 3])
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A*x <= a)
        model.addConstraint(2 <= B * x + D * y <= b)
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)

        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y[0] + 2 * y[1]


        s = CyClpSimplex(model)

        s.primal()
        sol = np.concatenate((s.primalVariableSolution['x'],
                              s.primalVariableSolution['y']))
        self.assertTrue((abs(sol -
                        np.array([0.2, 2, 1.1, 0, 0.9]) ) <= 10**-6).all())

    def test_multiDim(self):
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray
        s = CyClpSimplex()
        x = s.addVariable('x', (5, 3, 6))
        s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5

        s += 0 <= x <= 1
        c = CyLPArray(range(18))

        s.objective = c * x[2, :, :] + c * x[0, :, :]
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue(abs(sol[0, 1, 0] - 1) <= 10**-6)
        self.assertTrue(abs(sol[2, 0, 3] - 1) <= 10**-6)

    def test_ArrayIndexing(self):
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray
        s = CyClpSimplex()
        x = s.addVariable('x', (5, 3, 6))
        s += 2 * x[2, :, 3].sum() + 3 * x[0, 1, :].sum() >= 5


        s += x[1, 2, [0, 3, 5]] - x[2, 1, np.array([1, 2, 4])] == 1
        s += 0 <= x <= 1
        c = CyLPArray(range(18))

        s.objective = c * x[2, :, :] + c * x[0, :, :]
        s.primal()
        sol = s.primalVariableSolution['x']
        self.assertTrue(abs(sol[1, 2, 0] - 1) <= 10**-6)
        self.assertTrue(abs(sol[1, 2, 3] - 1) <= 10**-6)
        self.assertTrue(abs(sol[1, 2, 5] - 1) <= 10**-6)

    def test_primalAndDualColumnSolutions(self):
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

        s.initialDualSolve()
        self.assertEqual(s.getStatusCode(), 0)
        self.assertEqual(round(s.objectiveValue, 4), 1.3000)

        primal_sol = s.primalVariableSolution
        dual_sol = s.dualVariableSolution

        # assert that primalColumnSolution and primalVariableSolution are equivalent
        # i.e. primalColumnSolution is single array containing all the primal variable solutions
        self.assertTrue((s.primalColumnSolution == np.concatenate([primal_sol['x'], primal_sol['y']])).all())

        # assert that dualColumnSolution and dualVariableSolution are equivalent
        # i.e. dualColumnSolution is single array containing all the dual variable solutions
        self.assertTrue((s.dualColumnSolution == np.concatenate([dual_sol['x'], dual_sol['y']])).all())

        self.assertTrue((abs(primal_sol['x'] - np.array([0.2, 2.0 , 1.1]) ) <= 10**-6).all())
        self.assertTrue((abs(primal_sol['y'] - np.array([0.0, 0.9]) ) <= 10**-6).all())

        self.assertTrue((abs(dual_sol['x'] - np.array([0.0, -2.0,  3.0]) ) <= 10**-6).all())
        self.assertTrue((abs(dual_sol['y'] - np.array([1.0, 0.0]) ) <= 10**-6).all())

if __name__ == '__main__':
    unittest.main()
