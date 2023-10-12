# cython: embedsignature=True

cdef void RunGenerateCuts(void *ptr, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info):
    (<CyCglCutGeneratorBase>(ptr)).generateCuts(si, cs, info)

cdef CppCglCutGenerator* RunCglClone(void *ptr):
    return (<CyCglCutGeneratorBase>(ptr)).clone()


cdef class CyCglCutGeneratorBase:
    def __init__(self):
        Py_INCREF(self)
        self.CppSelf = new CppCglCutGeneratorBase(
            <cpy_ref.PyObject*>self,
            <runGenerateCuts_t>RunGenerateCuts,
            <runCglClone_t>RunCglClone)

    def __dealloc__(self):
        Py_DECREF(self)
        del self.CppSelf

    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info):
        raise Exception('CyCglCutGenerator.pyx: generateCuts must' \
                        ' be implemented.')

    cdef CppCglCutGenerator* clone(self):
        cdef CppCglCutGenerator* ret =  \
                <CppCglCutGenerator*> new CppCglCutGeneratorBase(
                                            <cpy_ref.PyObject*>self,
                                            <runGenerateCuts_t>RunGenerateCuts,
                                            <runCglClone_t>RunCglClone)
        return ret

#    cdef CyClpSimplex.CppIClpSimplex* model(self):
#        return self.CppSelf.model()
#
#    cdef void setModel(self, CyClpSimplex.CppIClpSimplex* m):
#        self.CppSelf.setModel(m)
#
#    property nRows:
#        def __get__(self):
#            return self.CppSelf.model().getNumRows()
#
#    property nCols:
#        def __get__(self):
#            return self.CppSelf.model().getNumCols()
#
