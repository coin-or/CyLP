# cython: embedsignature=True


cimport CyCglCutGeneratorBase

cdef class CyCglCutGeneratorBase(CyCglCutGeneratorBase):
    def __init__(self, pivotMethodObject):
        CyCglCutGeneratorBase.__init__(self)
        self.pivotMethodObject = pivotMethodObject

#    cdef pivotColumn(self, CppCoinIndexedVector* updates,
#                     CppCoinIndexedVector* spareRow1, CppCoinIndexedVector* spareRow2,
#                     CppCoinIndexedVector* spareCol1, CppCoinIndexedVector* spareCol2):
#        cyupdates = CyCoinIndexedVector()
#        cyupdates.setCppSelf(updates)
#        cyspareRow1 = CyCoinIndexedVector()
#        cyspareRow1.setCppSelf(spareRow1)
#        cyspareRow2 = CyCoinIndexedVector()
#        cyspareRow2.setCppSelf(spareRow2)
#        cyspareCol1 = CyCoinIndexedVector()
#        cyspareCol1.setCppSelf(spareCol1)
#        cyspareCol2 = CyCoinIndexedVector()
#        cyspareCol2.setCppSelf(spareCol2)
#        return self.pivotMethodObject.pivotColumn(cyupdates,
#                                    cyspareRow1, cyspareRow2,
#                                    cyspareCol1, cyspareCol2)
#

    cdef generateCuts(self, CppOsiSolverInterface *si,
                                     CppOsiCuts *cs,
                                     CppCglTreeInfo info):
        cysi =  CyOsiSolverInterface()
        cysi.setCppSelf(si)
        cycs = CyOsiCuts()
        cycs.setCppSelf(cs)
        cyinfo = CyCglTreeInfo()
        cyinfo.setCppSelf(info)
        self.cutGeneratorObject.generateCuts(cysi, cycs, cyinfo)

    cdef CyCglCutGenerator* clone(self):
        cdef CyCglCutGenerator* ret =  \
                <CyCglCutGenerator*> new CppCglCutGeneratorBase(
                            <cpy_ref.PyObject*>self,
                            <runGenerateCuts_t>RunGenerateCuts,
                            <runCglClone_t>RunCglClone)
        return ret


