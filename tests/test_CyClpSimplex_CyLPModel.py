import unittest
import inspect
import os
from os.path import join
import numpy as np

from CyLP.cy import CyClpSimplex

from CyLP.py.modeling.CyLPModel import CyLPModel, CyLPArray

currentFilePath = os.path.dirname(inspect.getfile(inspect.currentframe()))

class TestModel(unittest.TestCase):

    def test(self):
        model = CyLPModel()

        x = model.addVariable('x', 3)

        A = np.matrix([[1,2,3], [1,1,1]])
        b = CyLPArray([5, 3])

        model.addConstraint(A * x == b)
        model.addConstraint(x >= 0)
        
        model.objective = 1*x[0]  + 1*x[1] + 1.1 * x[2]

        # Solve it a first time
        s = CyClpSimplex(model)
        s.primal()
        sol = s.primalVariableSolution
        self.assertTrue((abs(sol - np.array([1,2,0]) ) <= 10**-6).all())
        # Add a cut
        s.addConstraint(x[0] >= 1.1)
        s.primal()
        self.assertTrue((abs(sol - np.array([1.1, 1.8, 0.1]) ) <= 10**-6).all())
        
        # Change the objective function
        c = CyLPArray([1, 10, 1.1])
        s.objective = c * x
        s.primal()
        self.assertTrue((abs(sol - np.array([2, 0, 1]) ) <= 10**-6).all())


    def test_multiVar(self):
        model = CyLPModel()

        x = model.addVariable('x', 3)
        y = model.addVariable('y', 2)

        A = np.matrix([[1., 2., 0],[1., 0, 1.]]) 
        B = np.matrix([[1., 0, 0], [0, 0, 1.]])
        D = np.matrix([[1., 2.],[0, 1]]) 
        a = CyLPArray([5, 2.5])
        b = CyLPArray([4.2, 3]) 
        x_u= CyLPArray([2., 3.5])

        model.addConstraint(A*x <= a)
        model.addConstraint(2 <= B * x + D * y <= b)
        model.addConstraint(y >= 0)
        model.addConstraint(1.1 <= x[1:3] <= x_u)
        
        c = CyLPArray([1., -2., 3.])
        model.objective = c * x + 2 * y[0] + 2 * y[1]
        
        
        s = CyClpSimplex(model)
        
        s.primal()
        sol = s.primalVariableSolution
        self.assertTrue((abs(sol - 
                        np.array([0.2, 2, 1.1, 0, 0.9]) ) <= 10**-6).all())


        s.addConstraint(x[2] + y[1] >= 2.1)
        s.primal()
        sol = s.primalVariableSolution
        self.assertTrue((abs(sol - 
                        np.array([0, 2, 1.1, 0, 1]) ) <= 10**-6).all())
        

if __name__ == '__main__':
    unittest.main()

