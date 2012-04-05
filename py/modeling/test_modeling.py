import unittest
import numpy as np
from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray, getCoinInfinity

inf = getCoinInfinity()

class TestModeling(unittest.TestCase):

    def setUp(self):
        self.model = CyLPModel()
        model = self.model
        
        self.x = model.addVariable('x', 5)
        self.y = model.addVariable('y', 4)
        self.z = model.addVariable('z', 5)
        
        
        self.b = CyLPArray([3.1, 4.2])
        
        self.A = np.matrix([[1, 2, 3, 3, 4], [3, 2, 1, 2, 5]])
        self.B = np.matrix([[0, 0, 0, -1.5, -1.5], [-3, 0, 0, -2,0 ]])
        self.D = np.matrix([[1, 3, 1, 4], [2, 1,3, 5]])


    def test_bound1(self):
        model = self.model
        x = self.x

        model.addConstraint(x[2:4] >= 3)
        self.assertTrue((x.lower == np.array([-inf, -inf, 3, 3, -inf])).all())
        self.assertTrue((x.upper == np.array(5* [inf])).all())
    
    def test_bound2(self):
        model = self.model
        x = self.x

        model.addConstraint(x[2:4] <= 3)
        self.assertTrue((x.lower == np.array(5 * [-inf])).all())
        self.assertTrue((x.upper == np.array([inf, inf, 3, 3, inf])).all())
    
    def test_bound3(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(1.1 <= x[2:4] <= b)
        self.assertTrue((x.lower == np.array(
                            [-inf, -inf, 1.1, 1.1, -inf])).all())
        self.assertTrue((x.upper == np.array(
                            [inf, inf, b[0], b[1], inf])).all())
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((vu[:5] == np.array(
                                    [inf, inf, b[0], b[1], inf])).all())
        self.assertTrue((vl[:5] == np.array(
                                    [-inf, -inf, 1.1, 1.1, -inf])).all())
                
    def test_bound4(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(x[2:4] == b)
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((vu[:5] == np.array(
                                    [inf, inf, b[0], b[1], inf])).all())
        self.assertTrue((vl[:5] == np.array(
                                    [-inf, -inf, b[0], b[1], -inf])).all())
                
    def test_constraint_single1(self):
        model = self.model
        x = self.x

        model.addConstraint(1.1 <= x[0] - 3 * x[1:3] + 2 * x[2:5] <= 4.5)
        cons = model.constraints[0]
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((m.todense() == np.array(
                            [1, -3, -1, 2, 2])).all())
    
    def test_constraint_single2(self):
        model = self.model
        x = self.x

        model.addConstraint(-x[1] + 3 * x[1:3] + None * x >= 4.5)
        cons = model.constraints[0]
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((m.todense() == np.array(
                            [0, 2, 3, 0, 0])).all())
        self.assertTrue((cons.lower == np.array(
                            [4.5])).all())
        self.assertTrue((cons.upper == np.array(
                            [inf])).all())
        

    def test_constraint_single3(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(-3 * x[1] + b * x[1:3]  >= 4.5)
        cons = model.constraints[0]
        m, cl, cu, vl, vu  = model.makeMatrices()
        self.assertTrue((abs(m.todense()-np.array(
                            [0, 0.1, 4.2, 0, 0])) < 0.000001).all())

    def test_constraint_1(self):
        model = self.model
        x = self.x
        b = self.b

        model.addConstraint(2 <= -x <= 4.5)
        
        cons = model.constraints[0]
        m, cl, cu, vl, vu = model.makeMatrices()
        self.assertTrue((abs(m.todense()-np.array(
                            5 * [-1])) < 0.000001).all())

        #self.assertEqual(self.c[4], 3.3)
    

if __name__ == '__main__':
    unittest.main()

