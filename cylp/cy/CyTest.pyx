import sys
from time import perf_counter
from cylp.cy.CyClpSimplex cimport CyClpSimplex
from cylp.cy.CyDantzigPivot cimport CyDantzigPivot
from cylp.cy.CyPEPivot cimport CyPEPivot


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

    start = perf_counter()
    s.primal()
    print('Exec time: ',  perf_counter() - start)
    return s.objectiveValue
