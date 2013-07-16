# cython: embedsignature=True


cimport CyCutGeneratorPythonBase


cdef class CyCutGeneratorPythonBase(CyCglCutGeneratorBase):
    def __init__(self, cutGeneratorObject):
        CyCglCutGeneratorBase.__init__(self)
        self.cutGeneratorObject = cutGeneratorObject
        print id(self), 'self.cutGeneratorObject set'
        print self.cutGeneratorObject.generateCuts(1, 2, 3)
        print 'tested'


    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info):
        print 'cy too'
        cysi =  CyOsiSolverInterface()
        cysi.setCppSelf(si)
        cycs = CyOsiCuts()
        cycs.setCppSelf(cs)
        cyinfo = CyCglTreeInfo()
        cyinfo.setCppSelf(&info)
        print 'going to python'

        print 'in python: ', id(self)
        cuts = self.cutGeneratorObject.generateCuts(cysi, cycs, cyinfo)
        print 'python returned'
        for cut in cuts:
            if cut.isRange:
                cycs.addColumnCut(cut)
            else:
                cycs.addRowCut(cut)

    cdef CppCglCutGenerator* clone(self):
        cdef CppCglCutGenerator* ret =  \
                <CppCglCutGenerator*> new CppCglCutGeneratorBase(
                            <cpy_ref.PyObject*>self,
                            <runGenerateCuts_t>RunGenerateCuts,
                            <runCglClone_t>RunCglClone)
        return ret


