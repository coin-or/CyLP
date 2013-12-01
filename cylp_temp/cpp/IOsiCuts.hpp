#include "Python.h"
#include <iostream>
using namespace std;

#include "OsiCuts.hpp"
#include "OsiSolverInterface.hpp"


class CppOsiCuts : public OsiCuts
{
public:

    void addColumnCut(int size, int* lowerBoundInds, double* lowerBoundElements,
                        int* upperBoundInds, double* upperBoundElements);
    void addRowCut(int size, int* indices, double* elements,
                   double lowerBound, double upperBound);

    void eraseRowCut(int i);
};


