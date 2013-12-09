import unittest
import numpy as np
from cylp.cy import CyClpSimplex

class TestCyCoinIndexedVector(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 10.5, -11.3, 100, -50.5, 20], dtype=np.double)
        self.a2 = np.array([1000, 10.5, -11.3, 100, -50.5, 20], dtype=np.double)
        self.s = CyClpSimplex()
    
#    def test_gen(self):
#        w = np.array([3], dtype=np.int32)
#        self.assertEqual(self.s.argWeightedMax(self.a, w, 0.1), 5)
#    
#    def test_empty(self):
#        w = np.array([], dtype=np.int32)
#        self.assertEqual(self.s.argWeightedMax(np.array([]), w, 0.1), 0)
#    
#    def test_first(self):
#        w = np.array([0, 2], dtype=np.int32)
#        self.assertEqual(self.s.argWeightedMax(self.a, w, 99), 3)
#        self.assertEqual(self.s.argWeightedMax(self.a, w, 100), 0)

    def test_argMax4_1(self):
        w_ind = np.array([0, 2, 5], dtype=np.int32)
        self.assertEqual(self.s.argWeightedMax(self.a, 0, 100, w_ind), 5)

    def test_argMax4_2(self):
        w_ind = np.array([0, 2, 5], dtype=np.int32)
        w = np.array([1.5, -10, 4], dtype=np.double)
        self.assertEqual(self.s.argWeightedMax(self.a, 0, w, w_ind), 2)
    
    def test_argMax4_3(self):
        w_ind = np.array([0, 8, 21], dtype=np.int32)
        a_ind = np.array([2, 5, 8, 10, 20, 21] , dtype=np.int32)
        self.assertEqual(self.s.argWeightedMax(self.a, a_ind, 5.1, w_ind), 5)

    def test_argMax4_4(self):
        w_ind = np.array([0, 8, 21], dtype=np.int32)
        w = np.array([100, -10, 4], dtype=np.double)
        a_ind = np.array([2, 5, 8, 10, 20, 21] , dtype=np.int32)
        self.assertEqual(self.s.argWeightedMax(self.a, a_ind, w, w_ind), 2)
    
    def test_argMax_5(self):
        w_ind = np.array([2, 7, 100], dtype=np.int32)
        w = np.array([100, -10, 4], dtype=np.double)
        a_ind = np.array([2, 5, 8, 10, 20, 21] , dtype=np.int32)
        self.assertEqual(self.s.argWeightedMax(self.a2, a_ind, 10, w_ind), 0)


if __name__ == '__main__':
    unittest.main()

