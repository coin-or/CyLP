import sys
from time import clock
from CyLP.cy.CyClpSimplex cimport CyClpSimplex
from CyLP.cy.CyDantzigPivot cimport CyDantzigPivot
from CyLP.cy.CyPEPivot cimport CyPEPivot


def CySolve(fileName, method):
    cdef CyClpSimplex s = CyClpSimplex()
    s.readMps(fileName, 0, 0)

    cdef CyDantzigPivot dpivot
    cdef CyPEPivot ppivot

    if method == 'd':
        dpivot = CyDantzigPivot(s)
        #s.setPerturbation(50)
        s.setPrimalColumnPivotAlgorithm(dpivot.CppSelf)
    elif method == 'p':
        ppivot = CyPEPivot(s)
        s.setPrimalColumnPivotAlgorithm(ppivot.CppSelf)

    start = clock()
    s.primal()
    print 'Exec time: ',  clock() - start
    return s.objectiveValue
