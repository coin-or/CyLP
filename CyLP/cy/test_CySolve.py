import unittest
import os
import inspect
from os.path import join
from CyLP.cy.CyTest import CySolve
currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

adlittleSol = 225494.963162


class TestPySolve(unittest.TestCase):

    def test_dantzig(self):
        objval = CySolve(join(currentFilePath, '../input/netlib/adlittle.mps'),
                       'd')
        self.assertAlmostEqual(objval, adlittleSol, 6)


if __name__ == '__main__':
    unittest.main()
