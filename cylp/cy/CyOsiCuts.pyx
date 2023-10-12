import scipy
from scipy.sparse import csr_matrix
from cylp.py.modeling.CyLPModel import CyLPModel
cimport numpy as np
import numpy as np

cdef class CyOsiCuts:
    'CyOsiCuts documentation'
    def __cinit__(self):
        self.CppSelf = new CppOsiCuts()

    cdef setCppSelf(self, CppOsiCuts* s):
        del self.CppSelf
        self.CppSelf = s

    def printCuts(self):
        self.CppSelf.printCuts()

    property numberOfRowCuts:
        def __get__(self):
            return self.CppSelf.sizeRowCuts()

    property numberOfColumnCuts:
        def __get__(self):
            return self.CppSelf.sizeColCuts()

    property numberOfCuts:
        def __get__(self):
            return self.CppSelf.sizeCuts()

    def addColumnCut(self, cut, cyLpModel):
        '''
        Add ``cut`` to cuts. ``cut`` is a CyLPExpr
        and ``cut.isRange`` is ``True``.
        '''
        m = CyLPModel()
        for var in cyLpModel.variables:
            m.addVariable(var.name, var.dim, var.isInt)
        x = m.getVarByName('x')
        m += cut
        mat, cl, cu, vl, vu = cyLpModel.makeMatrices()

        inds = np.arange(len(vl), dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] vl_inds = inds
        cdef np.ndarray[np.double_t, ndim=1] vl_data = vl
        cdef np.ndarray[np.int32_t, ndim=1] vu_inds = inds
        cdef np.ndarray[np.double_t, ndim=1] vu_data = vu

        self.CppSelf.addColumnCut(len(vl),
                                  <int*>vl_inds.data,
                                  <double*>vl_data.data,
                                  <int*>vu_inds.data,
                                  <double*>vu_data.data)

    def addRowCut(self, cut, cyLpModel):
        '''
        Add ``cut`` to cuts. ``cut`` is a CyLPExpr
        and ``cut.isRange`` is ``False``.
        '''
        m = CyLPModel()
        for var in cyLpModel.variables:
            m.addVariable(var.name, var.dim, var.isInt)
        m += cut
        mat, cl, cu, vl, vu = m.makeMatrices()

        cdef np.ndarray[np.int32_t, ndim=1] row_inds
        cdef np.ndarray[np.double_t, ndim=1] row_data

        for nr in xrange(mat.shape[0]):
            row = mat[nr, :]
            row_inds = row.indices
            row_data = row.data
            assert len(row_inds) == len(row_data)
            self.CppSelf.addRowCut(len(row_inds),
                                  <int*>row_inds.data,
                                  <double*>row_data.data,
                                  cl[nr], cu[nr])


