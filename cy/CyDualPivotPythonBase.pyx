# cython: embedsignature=True


cimport CyDualPivotPythonBase

cdef class CyDualPivotPythonBase(CyClpDualRowPivotBase):
    def __init__(self, dualPivotMethodObject):
        CyClpDualRowPivotBase.__init__(self)
        self.dualPivotMethodObject = dualPivotMethodObject

    cdef pivotRow(self):
        return self.dualPivotMethodObject.pivotRow()

    cdef CyClpDualRowPivot* clone(self, bint copyData):
        cdef CyClpDualRowPivot* ret =  \
                <CyClpDualRowPivot*> new CppClpDualRowPivotBase(
                            <cpy_ref.PyObject*>self,
                            <runPivotRow_t>RunPivotRow,
                            <runDualPivotClone_t>RunDualPivotClone,
                            <runUpdateWeights_t>RunUpdateWeights,
                            <runUpdatePrimalSolution_t>RunUpdatePrimalSolution)
        return ret

    cdef double updateWeights(self, CppCoinIndexedVector* inp,
                                  CppCoinIndexedVector* spare,
                                  CppCoinIndexedVector* spare2,
                                  CppCoinIndexedVector* updatedColumn):
        print 'CyUpdateWeights'
        cyinp = CyCoinIndexedVector()
        cyinp.setCppSelf(inp)
        cyspare = CyCoinIndexedVector()
        cyspare.setCppSelf(spare)
        cyspare2 = CyCoinIndexedVector()
        cyspare2.setCppSelf(spare2)
        cyupdatedColumn = CyCoinIndexedVector()
        cyupdatedColumn.setCppSelf(updatedColumn)
        ret = self.dualPivotMethodObject.updateWeights(cyinp, cyspare, cyspare2, cyupdatedColumn)
        print 'CyDualPivotPythonBase alpha: ', ret
        return ret

    cdef void updatePrimalSolution(self,
                                   CppCoinIndexedVector* inp,
                                   double theta,
                                   double * changeInObjective):
        print 'CyUpdatePrimalSolution'
        cyinp = CyCoinIndexedVector()
        cyinp.setCppSelf(inp)
        change = self.dualPivotMethodObject.updatePrimalSolution(cyinp, theta)
        changeInObjective[0] = change
