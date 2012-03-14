import unittest
import numpy as np
from CyLP.cy import CyCoinModel, CyClpSimplex

class TestCyCoinModel(unittest.TestCase):

    def test(self):
        m = CyCoinModel()
        
        m.addVariable(3, np.array(
                        [0, 1, 2], np.int32),
                        np.array([1., 1., 1.], np.double), 0, 10, 5)

        m.addVariable(2, np.array(
                        [1,2], np.int32),
                        np.array([5, 2.], np.double), 0, 10, 2)

        # Add bound for the three constraints (we have two variables)
        m.setConstraintLower(0, 2.3)
        m.setConstraintLower(1, 4.5)
        m.setConstraintLower(0, 1.5)
        
        # Add a 4th constraint
        m.addConstraint(2, 
                            np.array([0, 1], np.int32), 
                            np.array([1., 1.], np.double), 2, 7)
        
        s = CyClpSimplex()
        # Load the problem from the CyCoinModel
        s.loadProblemFromCyCoinModel(m)
        
        s.primal()
         
        self.assertAlmostEqual(s.objectiveValue, 8.7, 7)
        
if __name__ == '__main__':
    unittest.main()

