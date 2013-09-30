import unittest
import os
import inspect
from os.path import join
from CyLP.cy.CySolve import CySolve


problem = os.path.join(os.path.dirname(__file__), '..', 'input',
                       'netlib', 'adlittle.mps')

adlittleSol = 225494.963162


class TestPySolve(unittest.TestCase):

    def test_dantzig(self):
        objval = CySolve(problem, 'd')
        self.assertAlmostEqual(objval, adlittleSol, 6)


if __name__ == '__main__':
    unittest.main()
