# cython: embedsignature=True

cdef class CyCutGeneratorPythonBase(CyCglCutGeneratorBase):
    def __init__(self, cutGeneratorObject, cyLPModel=None):
        CyCglCutGeneratorBase.__init__(self)
        self.cutGeneratorObject = cutGeneratorObject
        Py_INCREF(cutGeneratorObject)
        self.cyLPModel = cyLPModel

    def __dealloc__(self):
        Py_DECREF(self.cutGeneratorObject)

    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info):
        cysi =  CyOsiSolverInterface()
        cysi.setCppSelf(si)
        cycs = CyOsiCuts()
        cycs.setCppSelf(cs)

        cyinfo = CyCglTreeInfo()
        cyinfo.setCppSelf(&info)
        cuts = self.cutGeneratorObject.generateCuts(cysi, cyinfo)
        if type(cuts) is not list:
            cuts = [cuts]
        for cut in cuts:
            # Getting a CyLPConstraint
            isRange = cut.evaluate('cut').isRange
            if isRange:
                cycs.addColumnCut(cut, self.cyLPModel)
            else:
                cycs.addRowCut(cut, self.cyLPModel)

    cdef CppCglCutGenerator* clone(self):
        cdef CppCglCutGenerator* ret =  \
                <CppCglCutGenerator*> new CppCglCutGeneratorBase(
                            <cpy_ref.PyObject*>self,
                            <runGenerateCuts_t>RunGenerateCuts,
                            <runCglClone_t>RunCglClone)
        return ret


