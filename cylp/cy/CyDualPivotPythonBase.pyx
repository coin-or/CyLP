# cython: embedsignature=True

cdef class CyDualPivotPythonBase(CyClpDualRowPivotBase):
    def __init__(self, dualPivotMethodObject):
        CyClpDualRowPivotBase.__init__(self)
        Py_INCREF(dualPivotMethodObject)
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
        cyinp = CyCoinIndexedVector()
        cyinp.setCppSelf(inp)
        cyspare = CyCoinIndexedVector()
        cyspare.setCppSelf(spare)
        cyspare2 = CyCoinIndexedVector()
        cyspare2.setCppSelf(spare2)
        cyupdatedColumn = CyCoinIndexedVector()
        cyupdatedColumn.setCppSelf(updatedColumn)
        return self.dualPivotMethodObject.updateWeights(cyinp, cyspare, cyspare2, cyupdatedColumn)

    cdef void updatePrimalSolution(self,
                                   CppCoinIndexedVector* inp,
                                   double theta,
                                   np.ndarray[np.double_t,ndim=1] changeInObjective):
        cyinp = CyCoinIndexedVector()
        cyinp.setCppSelf(inp)
        change = self.dualPivotMethodObject.updatePrimalSolution(cyinp, theta, changeInObjective)
        changeInObjective[0] = change
