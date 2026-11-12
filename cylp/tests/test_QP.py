import unittest
import importlib
import inspect
import os
import sys
from os.path import join
import numpy as np

from cylp.cy import CyClpSimplex
from cylp.py.pivots import WolfePivot

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestQP(unittest.TestCase):
    def test_1(self):
        """simplest QP test"""
        s = CyClpSimplex()
        self.assertEqual(s.readMps(join(currentFilePath, '../input/hs35.qps')), 0)
        #self.assertTrue(abs(cbcModel.objectiveValue - 3089.0) < 10 ** -6)

        #print(s.Hessian.todense())

        p = WolfePivot(s)
        s.setPivotMethod(p)

        s.primal()
        print(s.primalVariableSolution)
        print(s.objectiveValue)

    def test_deprecated(self):
        'cylp.py.QP and the Givens functions that it uses are deprecated'
        for name in list(sys.modules):
            if name == 'cylp.py.QP' or name.startswith('cylp.py.QP.'):
                del sys.modules[name]
        with self.assertWarns(DeprecationWarning):
            importlib.import_module('cylp.py.QP')

        from cylp.py.utils import util
        with self.assertWarns(DeprecationWarning):
            util.givens(2, 0, 1, 1., 1.)

if __name__ == '__main__':
    unittest.main()

