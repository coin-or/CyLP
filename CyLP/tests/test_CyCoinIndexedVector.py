import unittest
import numpy as np
from cylp.cy import CyCoinIndexedVector

class TestCyCoinIndexedVector(unittest.TestCase):

    def setUp(self):
        self.c = CyCoinIndexedVector()
        self.c.reserve(5)

    def test_basic(self):
        c = self.c
        c[1] = 3.1
        c[4] = -2
        self.assertTrue((abs(c.elements - np.array([0., 3.1, 0., 0., -2.])) < 
                                                            10 ** -8).all())
        self.assertTrue((c.indices - np.array([1,4]) < 10 ** -8).all())
        self.assertTrue(c.nElements == 2)
    
    def test_number_number(self):
        self.c.clear()
        self.c[2] = -1.2
        self.assertEqual(self.c[1], 0)
        self.assertEqual(self.c[2], -1.2)
        self.c.clear()
        self.assertEqual(self.c[2], 0)
        self.c[2] = 1.5
        self.assertEqual(self.c[2], 1.5)

    def test_slice_list_number(self):
        l = [1, 3, 4]
        self.c[l] = 5.5
        self.assertEqual(self.c[1], 5.5)
        self.assertEqual(self.c[3], 5.5)
        self.assertEqual(self.c[4], 5.5)

    def test_slice_list_list(self):
        l = [1, 3, 4]
        val = [1.1, 2.2, 3.3]
        self.c[l] = val
        self.assertEqual(self.c[1], 1.1)
        self.assertEqual(self.c[3], 2.2)
        self.assertEqual(self.c[4], 3.3)
    
    def test_slice_list_array(self):
        l = [1, 3, 4]
        val = np.array([1.1, 2.2, 3.3])
        self.c[l] = val
        self.assertEqual(self.c[1], 1.1)
        self.assertEqual(self.c[3], 2.2)
        self.assertEqual(self.c[4], 3.3)
    
    def test_slice_array_number(self):
        l = np.array([1, 3, 4])
        val = 5.5
        self.c[l] = val
        self.assertEqual(self.c[1], 5.5)
        self.assertEqual(self.c[3], 5.5)
        self.assertEqual(self.c[4], 5.5)

    def test_slice_array_list(self):
        l = np.array([1, 3, 4])
        val = [1.1, 2.2, 3.3]
        self.c[l] = val
        self.assertEqual(self.c[1], 1.1)
        self.assertEqual(self.c[3], 2.2)
        self.assertEqual(self.c[4], 3.3)

    def test_slice_array_array(self):
        l = np.array([1, 3, 4])
        val = np.array([1.1, 2.2, 3.3])
        self.c[l] = val
        self.assertEqual(self.c[1], 1.1)
        self.assertEqual(self.c[3], 2.2)
        self.assertEqual(self.c[4], 3.3)
    
    def test_reAssign(self):
        c = self.c
        c[3] = 5
        l = [1, 3, 4]
        c[l] = [1, 1, 1]
        self.assertEqual(self.c[0], 0)
        self.assertEqual(self.c[1], 1)
        self.assertEqual(self.c[3], 1)

if __name__ == '__main__':
    unittest.main()

