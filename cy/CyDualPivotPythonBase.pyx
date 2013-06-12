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
        #TODO: change cppCoinIndexedVector
        return 0

    cdef void updatePrimalSolution(self,
                                   CppCoinIndexedVector* inp,
                                   double theta,
                                   double * changeInObjective):
        pass
