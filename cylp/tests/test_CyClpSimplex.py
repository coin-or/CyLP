import unittest
import inspect
import os
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex, CyCoinPackedMatrix
from cylp.cy.CyClpSimplex import statusString
from cylp.py.modeling.CyLPModel import CyLPArray

from cylp.py.pivots import PositiveEdgePivot
from cylp.py.pivots import LIFOPivot
from cylp.py.pivots import MostFrequentPivot
from cylp.py.pivots import DantzigPivot

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestCyClpSimplex(unittest.TestCase):

    def setUp(self):
        self.s = CyClpSimplex()
        self.assertEqual(
            self.s.readMps(join(currentFilePath, '../input/p0033.mps')), 0)

    def test_PE(self):
        #pivot = PositiveEdgePivot(self.s)
        self.s.setPivotMethod(PositiveEdgePivot(self.s))
        self.s.primal()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_Dantzig(self):
        #pivot = DantzigPivot(self.s)
        self.s.setPivotMethod(DantzigPivot(self.s))
        self.s.primal()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_LIFO(self):
        #pivot = LIFOPivot(self.s)
        self.s.setPivotMethod(LIFOPivot(self.s))
        self.s.primal()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_MostFrequent(self):
        #pivot = MostFrequentPivot(self.s)
        self.s.setPivotMethod(MostFrequentPivot(self.s))
        self.s.primal()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_initialSolve(self):
        self.s.initialSolve()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_initialPrimalSolve(self):
        self.s.initialPrimalSolve()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_initialDualSolve(self):
        self.s.initialDualSolve()
        self.assertEqual(round(self.s.objectiveValue, 4), 2520.5717)

    def test_statusString(self):
        # Clp's status is -1 until the problem is solved
        self.assertEqual(self.s.getStatusString(), 'unknown')
        self.assertEqual(statusString(10), 'stopped, try another algorithm')
        self.assertEqual(statusString(42), 'unknown status 42')

    def test_QPWithoutBoundedRows(self):
        "Clp needs a QP to have a constraint with a finite bound"
        def qp(constraint=None):
            s = CyClpSimplex()
            x = s.addVariable('x', 2)
            if constraint:
                s += constraint(x)
            s += 0 <= x <= 10
            s.objective = CyLPArray([-3., -3.]) * x
            # x0^2 + x0 x1 + x1^2, given as one triangle
            s.loadQuadraticObjective(CyCoinPackedMatrix(True,
                    np.array([0, 0, 1], np.int32),
                    np.array([0, 1, 1], np.int32), np.array([2., 1., 2.])))
            return s

        # Without one, Clp's primal gave up and initialSolve aborted
        for constraint in (None, lambda x: -1e30 <= x[0] + x[1] <= 1e30):
            s = qp(constraint)
            for solve in (s.primal, s.initialSolve, s.initialPrimalSolve,
                          s.initialDualSolve):
                self.assertRaises(ValueError, solve)

        # A constraint that can't be active is enough
        for solve in ('primal', 'initialSolve'):
            s = qp(lambda x: x.sum() <= 1e10)
            self.assertEqual(getattr(s, solve)(), 'optimal')
            self.assertTrue((abs(s.primalVariableSolution['x'] -
                                 np.array([1, 1])) <= 10**-6).all())

    def test_primalVariableSolutionAll(self):
        self.s.initialSolve()
        n = self.s.nVariables
        x = self.s.primalVariableSolution
        sol = self.s.primalVariableSolutionAll
        self.assertEqual(len(sol), n + self.s.nConstraints)
        self.assertTrue((sol[:n] == x).all())
        # The slacks are the constraints' activities
        self.assertTrue((abs(sol[n:] - self.s.coefMatrix @ x) <= 10**-6).all())

    def test_direction(self):
        self.assertEqual(self.s.optimizationDirection, 'min')
        self.s.optimizationDirection = 'max'
        self.assertEqual(self.s.optimizationDirection, 'max')

    def test_primalEmptyModel(self):
        # A model that was never loaded has no objective or matrix
        s = CyClpSimplex()
        self.assertEqual(s.primal(), 'optimal')



if __name__ == '__main__':
    unittest.main()

