import scipy
from scipy.sparse import csr_matrix
cimport CyOsiCuts
from CyLP.py.modeling.CyLPModel import CyLPModel

cdef class CyOsiCuts:
    'CyOsiCuts documentation'
    def __cinit__(self):
        self.CppSelf = new CppOsiCuts()

    cdef setCppSelf(self, CppOsiCuts* s):
        del self.CppSelf
        self.CppSelf = s

    def addColumnCut(self, cut):
        '''
        Add ``cut`` to cuts. ``cut`` is a CyLPConstraint
        and ``cut.isRange`` is ``True``.
        '''
        # CyLPConstraint's lower and upper are dense vectors
        # First convert them to sparse
        lower = csr_matrix(cut.lower)
        upper = csr_matrix(cut.upper)

        cdef np.ndarray[np.int32_t, ndim=1] lower_inds = lower.indices
        cdef np.ndarray[np.double_t, ndim=1] lower_data = lower.data
        cdef np.ndarray[np.int32_t, ndim=1] upper_inds = upper.indices
        cdef np.ndarray[np.double_t, ndim=1] upper_data = upper.data

        self.CppSelf.addColumnCut(cut.upper.shape[1],
                                  <int*>lower_inds.data,
                                  <double*>lower_data.data,
                                  <int*>upper_inds.data,
                                  <double*>upper_data.data)

    def addRowCut(self, cut, cyLpModel):
        '''
        Add ``cut`` to cuts. ``cut`` is a CyLPConstraint
        and ``cut.isRange`` is ``False``.
        '''
        m = CyLPModel()
        for var in m.vars:
            m.addVariable(var.name, var.dim, var.isInt)
        m += cut
        mat, cl, cu, vl, vu = m.makeMatrices()

        cdef np.ndarray[np.int32_t, ndim=1] row_inds
        cdef np.ndarray[np.double_t, ndim=1] row_data

        for nr in xrange(mat.shape[0]):
            row = mat[nr, :]
            row_inds = row.indices
            row_data = row.data
            self.CppSelf.addRowCut(mat.shape[1],
                                  <int*>row_inds.data,
                                  <double*>row_data.data,
                                  cl[nr], cu[nr])


