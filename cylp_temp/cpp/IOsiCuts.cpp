#include "IOsiCuts.hpp"

void
CppOsiCuts::addColumnCut(int size, int* lowerBoundInds, double* lowerBoundElements,
                        int* upperBoundInds, double* upperBoundElements){
    OsiColCut cc;
    cc.setLbs(size, lowerBoundInds, lowerBoundElements);
    cc.setUbs(size, upperBoundInds, upperBoundElements);
    insert(cc);
}


void
CppOsiCuts::addRowCut(int size, int* indices, double* elements,
                   double lowerBound, double upperBound){
      OsiRowCut rc;
      rc.setRow(size, indices, elements);
      rc.setLb(lowerBound);
      rc.setUb(upperBound);
      insert(rc);
}


void CppOsiCuts::eraseRowCut(int i){
    OsiCuts::eraseRowCut(i);
}
