import unittest
import numpy as np

from cylp.py.utils.sparseUtil import csr_matrixPlus, csc_matrixPlus


class TestSpaseUtil(unittest.TestCase):


    def test_scale_scale(self):
        m = csc_matrixPlus(([1, 2, 6, 4, 5, 7, 3, 2],
                            ([0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 0, 2, 3, 0, 2, 3])),
                            shape=(3, 4), dtype=np.float)
        unscaledM = m.copy()
        row_scale = m.row_scale()
        col_scale = m.col_scale()
        m.col_scale(1/col_scale)
        m.row_scale(1/row_scale)
        assert(((m - unscaledM).todense() < 1.0e-7).all())

    def test_scale_unscale(self):
        m = csc_matrixPlus(([1, 2, 6, 4, 5, 7, 3, 2],
                            ([0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 0, 2, 3, 0, 2, 3])),
                            shape=(3, 4), dtype=np.float)
        unscaledM = m.copy()
        m.row_scale()
        m.col_scale()
        m.col_unscale()
        m.row_unscale()
        assert(((m - unscaledM).todense() < 1.0e-7).all())



if __name__ == '__main__':
    unittest.main()

