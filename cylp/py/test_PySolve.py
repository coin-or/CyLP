import unittest
import inspect
import os
from os.path import join
import numpy as np
from cylp.cy import CyClpSimplex
from .PySolve import solve


problem = os.path.join(os.path.dirname(__file__), '..', 'input',
                       'netlib', 'adlittle.mps')

adlittleSol = 225494.963162

class TestPySolve(unittest.TestCase):

    def test_pe(self):
        objval = solve(problem, 'p')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_dantzig(self):
        objval = solve(problem, 'd')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_lifo(self):
        objval = solve(problem, 'l')
        self.assertAlmostEqual(objval, adlittleSol, 6)

    def test_mf(self):
        objval = solve(problem, 'm')
        self.assertAlmostEqual(objval, adlittleSol, 6)


if __name__ == '__main__':
    unittest.main()

