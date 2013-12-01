import unittest
import numpy as np

from cylp.py.modeling import IndexFactory

class TestIndexFactory(unittest.TestCase):

    def setUp(self):
        inds = self.inds = IndexFactory()
        inds.addConst('Ax', 5)
        inds.addConst('Bx', 10)
        inds.addConst('Cx', 15)
        inds.addConst('Dx', 20)
        
        inds.addVar('x', 5)
        inds.addVar('a', 10)
        inds.addVar('b', 15)
        inds.addVar('c', 20)

    def test_c1(self):
        inds = self.inds
        
        self.assertTrue((inds.constIndex['Bx'] == np.arange(5, 15)).all())
        inds.removeConst('Ax')
        self.assertTrue((inds.constIndex['Bx'] == np.arange(0, 10)).all())
        self.assertTrue((inds.constIndex['Cx'] == np.arange(10, 25)).all())
        self.assertTrue((inds.constIndex['Dx'] == np.arange(25, 45)).all())
        self.assertTrue('Ax' not in inds.constIndex.keys())
        
    def test_c2(self):
        inds = self.inds
        
        self.assertTrue((inds.constIndex['Cx'] == np.arange(15, 30)).all())
        inds.removeConst('Bx')
        self.assertTrue((inds.constIndex['Ax'] == np.arange(0, 5)).all())
        self.assertTrue((inds.constIndex['Cx'] == np.arange(5, 20)).all())
        self.assertTrue((inds.constIndex['Dx'] == np.arange(20, 40)).all())
        self.assertTrue('Bx' not in inds.constIndex.keys())
    
    def test_v1(self):
        inds = self.inds
        
        self.assertTrue((inds.varIndex['x'] == np.arange(0, 5)).all())
        inds.removeVar('x')
        self.assertTrue(not inds.hasVar('x'))
        self.assertTrue((inds.varIndex['a'] == np.arange(0, 10)).all())
        self.assertTrue((inds.varIndex['b'] == np.arange(10, 25)).all())
        self.assertTrue((inds.varIndex['c'] == np.arange(25, 45)).all())
        
    def test_v2(self):
        inds = self.inds
        
        self.assertTrue((inds.varIndex['b'] == np.arange(15, 30)).all())
        inds.removeVar('a')
        self.assertTrue(not inds.hasVar('a'))
        self.assertTrue((inds.varIndex['x'] == np.arange(0, 5)).all())
        self.assertTrue((inds.varIndex['b'] == np.arange(5, 20)).all())
        self.assertTrue((inds.varIndex['c'] == np.arange(20, 40)).all())

    def test_reverse(self):
        inds = self.inds
        
        self.assertTrue(inds.reverseVarSearch(14) == ('a', 9))
        self.assertTrue(inds.reverseVarSearch(15) == ('b', 0))
        self.assertTrue(inds.reverseVarSearch(34) == ('c', 4))


if __name__ == '__main__':
    unittest.main()

