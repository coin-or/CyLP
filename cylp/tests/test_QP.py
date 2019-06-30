from __future__ import print_function
import unittest
import inspect
import os
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex
from cylp.py.pivots import WolfePivot

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestQP(unittest.TestCase):
    def test_1(self):
        """simplest QP test"""
        s = CyClpSimplex()
        s.readMps(join(currentFilePath, '../input/hs35.qps'))
        #self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)

        #print(s.Hessian.todense())

        p = WolfePivot(s)
        s.setPivotMethod(p)

        s.primal()
        print(s.primalVariableSolution)
        print(s.objectiveValue)

if __name__ == '__main__':
    unittest.main()

