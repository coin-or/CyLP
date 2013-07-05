import unittest
import inspect
import os
from os.path import join
import numpy as np
from CyLP.cy import CyClpSimplex
from PySolve import solve
currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

adlittleSol = 225494.963162

class TestPySolve(unittest.TestCase):

    def test_pe(self):
        objval = solve(join(currentFilePath, '../input/netlib/adlittle.mps'),
                       'p')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_dantzig(self):
        objval = solve(join(currentFilePath, '../input/netlib/adlittle.mps'),
                       'd')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_lifo(self):
        objval = solve(join(currentFilePath, '../input/netlib/adlittle.mps'),
                       'l')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_mf(self):
        objval = solve(join(currentFilePath, '../input/netlib/adlittle.mps'),
                       'm')
        self.assertAlmostEqual(objval, adlittleSol, 6)


if __name__ == '__main__':
    unittest.main()

