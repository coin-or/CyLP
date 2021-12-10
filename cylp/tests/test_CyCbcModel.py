import numpy as np
import unittest

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel


class TestModel(unittest.TestCase):

    def test_milp(self):
        
        # Scenario:

        # supplier 1
        capacity_1 = 100
        threshold_1 = 50
        cost_1 = 10

        # supplier 2
        capacity_2 = 100
        threshold_2 = 70
        cost_2 = 8

        # demand
        demand = 60

        
        # Model:

        model = CyLPModel()
                                                                # columns 
        supply_1 = model.addVariable('supply_1', 1)             # 0
        model += 0 <= supply_1 <= capacity_1
        supply_2 = model.addVariable('supply_2', 1)             # 1
        model += 0 <= supply_2 <= capacity_2

        switch_1 = model.addVariable('switch_1', 1, isInt=True) # 2
        model += 0 <= switch_1 <= 1
        switch_2 = model.addVariable('switch_2', 1, isInt=True) # 3
        model += 0 <= switch_2 <= 1

        # enforce threshold                                     # rows
        model += supply_1 - threshold_1 * switch_1 >= 0         # 0
        model += supply_1 - capacity_1 * switch_1 <= 0          # 1
        model += supply_2 - threshold_2 * switch_2 >= 0         # 2
        model += supply_2 - capacity_2 * switch_2 <= 0          # 3

        # demand
        model += supply_1 + supply_2 == demand                  # 4

        # minimise cost
        model.objective = cost_1 * supply_1 + cost_2 * supply_2


        clpModel = CyClpSimplex(model)
        cbcModel = clpModel.getCbcModel()
        cbcModel.solve()

        self.assertTrue(len(cbcModel.primalColumnSolution) == 4)
        #                                                          supply_1  supply_2  switch_1  switch_2
        self.assertTrue(np.isclose(cbcModel.primalColumnSolution, [60,       0,        1,        0       ]).all())

        self.assertTrue(len(cbcModel.dualRowSolution) == 5)
        #                                                       price
        self.assertTrue(np.isclose(cbcModel.dualRowSolution[4], 10   ))


if __name__ == '__main__':
    unittest.main()
