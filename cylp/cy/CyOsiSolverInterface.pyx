cdef extern from *:
    CppOsiClpSolverInterface* dynamic_cast_osi_2_clp "dynamic_cast<OsiClpSolverInterface*>" (CppOsiSolverInterface*)
    CppIClpSimplex* static_cast_clp_2_iclp "static_cast<IClpSimplex*>" (CppClpSimplex*)

cdef class CyOsiSolverInterface:
    'CyOsiSolverInterface documentation'
    def __cinit__(self):
        pass

    cdef setCppSelf(self, CppOsiSolverInterface* s):
        del self.CppSelf
        self.CppSelf = s

    property clpModel:
        def __get__(self):
            cdef CyClpSimplex s = CyClpSimplex()
            cdef CppOsiClpSolverInterface* si = dynamic_cast_osi_2_clp(self.CppSelf)
            cdef CppIClpSimplex* sp = static_cast_clp_2_iclp(si.getModelPtr())
            s.setCppSelf(sp)
            return s
