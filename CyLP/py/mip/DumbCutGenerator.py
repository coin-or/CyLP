import numpy as np
from CyLP.cy import CyClpSimplex
from CyLP.py.modeling import CyLPArray, CyLPModel

class DumbCutGenerator:
    def __init__(self, cyLPModel):
        self.cyLPModel = cyLPModel

    def generateCuts(self, x, y, z):
        print 'py cut called'
        m = self.cyLPModel
        x = m.getVarByName('x')
        return x[0] + x[1] >= 2

    def test():
        print 'test success'


if __name__ == '__main__':
    m = CyLPModel()

    x = m.addVariable('x', 2, isInt=True)

    A = np.matrix([[7., -2.],[0., 1], [2., -2]])
    b = CyLPArray([14, 3, 3])

    m += A * x <= b
    m += x >= 0

    c = CyLPArray([-4, 1])
    m.objective = c * x

    s = CyClpSimplex(m)
    cbcModel = s.getCbcModel()

    dc = DumbCutGenerator(m)
    cbcModel.addPythonCutGenerator(dc)

    cbcModel.branchAndBound()
