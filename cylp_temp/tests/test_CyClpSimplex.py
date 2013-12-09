import unittest
import inspect
import os
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex

from cylp.py.pivots import PositiveEdgePivot
from cylp.py.pivots import LIFOPivot
from cylp.py.pivots import MostFrequentPivot
from cylp.py.pivots import DantzigPivot

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestCyClpSimplex(unittest.TestCase):

    def setUp(self):
        self.s = CyClpSimplex()
        self.s.readMps(join(currentFilePath, '../input/p0033.mps'))

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


if __name__ == '__main__':
    unittest.main()

