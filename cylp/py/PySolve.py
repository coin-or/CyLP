from __future__ import print_function
import sys
from time import clock
import cProfile
import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.pivots import PositiveEdgePivot
from cylp.py.pivots.WolfePivot import WolfePivot
from cylp.py.pivots import LIFOPivot
from cylp.py.pivots import MostFrequentPivot
from cylp.py.pivots import DantzigPivot


def solve(filename, method):

    s = CyClpSimplex()
    s.readMps(filename)

    s.preSolve(feasibilityTolerance=10 ** -8)
    #s.useCustomPrimal(1)

    if method == 'd':
        pivot = DantzigPivot(s)
    elif method == 'l':
        pivot = LIFOPivot(s)
    elif method == 'm':
        pivot = MostFrequentPivot(s)
    elif method == 'p':
        pivot = PositiveEdgePivot(s)
    else:
        print('Unkown solution method.')
        sys.exit(1)

    s.setPivotMethod(pivot)

    #s.setPerturbation(50)

    start = clock()
    s.primal()
    print('Problem solved in %g seconds.' % (clock() - start))
    return s.objectiveValue


#    s.copyInIntegerInformation(np.array(s.nCols * [True], np.uint8))
#    #s.setInteger(100)
#
#    print("Solving relaxation")
#    cbcModel = s.getCbcModel()
#    from SimpleNodeCompare import SimpleNodeCompare
#    from CyCgl import CyCglGomory, CyCglClique, CyCglKnapsackCover
#    n = SimpleNodeCompare()
#    cbcModel.setNodeCompare(n)
##
#    gom = CyCglGomory(limit=150)
#    #gom.limit = 150
#    cbcModel.addCutGenerator(gom, name="Gomory")
#
#    #clq = CyCglClique()
#    #cbcModel.addCutGenerator(clq, name="Clique")
##
#
#    knap = CyCglKnapsackCover(maxInKnapsack=50)
#    cbcModel.addCutGenerator(knap, name="Knapsack")
#
#
#    cbcModel.branchAndBound()
#    print(cbcModel.primalVariableSolution)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage example (Dantzig pivot): PySolve file.mps d')
        sys.exit(1)
    solve(sys.argv[1], sys.argv[2])
    #cProfile.run('solve(sys.argv[1], sys.argv[2])')
