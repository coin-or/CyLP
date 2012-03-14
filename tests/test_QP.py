import unittest
import inspect
import os
from os.path import join
import numpy as np

from CyLP.cy import CyClpSimplex

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestQP(unittest.TestCase):
    def test_1(self):
        """simplest QP test"""
        s = CyClpSimplex()
        s.readMps(join(currentFilePath, '../input/hs35.qps'))
        #self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)
        print s.Hessian

if __name__ == '__main__':
    unittest.main()

