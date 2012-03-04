import unittest
import inspect
import os
from os.path import join
import numpy as np

from CyLP.cy import CyCoinMpsIO
from CyLP.cy.CyCoinMpsIO import getQpsExample

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))


class TestCyCoinMpsIO(unittest.TestCase):

    def test(self):
        problem = CyCoinMpsIO()
        problem.readMps(os.path.join(currentFilePath, '../input/hs268.qps'))
        self.assertEqual(problem.nVariables, 5)
        self.assertEqual(problem.nConstraints, 5)
        self.assertTrue([chr(s) for s in problem.constraintSigns] ==
                                        problem.nConstraints * ['G'])
        c = problem.matrixByRow

        self.assertTrue((abs(c.elements -
                        np.array([-1., -1., -1., -1., -1., 10.,
                                   10., -3., 5., 4., -8.,
                                   1., -2., -5., 3., 8., -1.,
                                   2., 5., -3., -4., -2.,
                                   3., -5., 1.])) <= 10 ** -8).all())

        self.assertTrue((abs(c.indices -
                        np.array([0, 1, 2, 3, 4, 0, 1, 2,
                                      3, 4, 0, 1, 2, 3, 4, 0,
                                      1, 2, 3, 4, 0, 1, 2,
                                       3, 4])) <= 10 ** -8).all())

        self.assertTrue((abs(c.vectorStarts -
                             np.array([0, 5, 10, 15, 20, 25])) <=
                                                        10 ** -8).all())

        self.assertTrue((abs(problem.rightHandSide -
                         np.array([-5., 20., -40., 11., -30.])) <=
                                                    10 ** -8).all())

        H = problem.Hessian.todense()
        self.assertTrue((abs(H -
            np.matrix([[20394., -24908., -2026., 3896., 658.],
                       [-24908., 41818., -3466., -9828., -372.],
                       [-2026., -3466., 3510., 2178., -348.],
                       [3896., -9828., 2178., 3030., -44.],
                       [658., -372., -348., -44., 54.]])) <= 10 ** -8).all())


if __name__ == '__main__':
    unittest.main()
