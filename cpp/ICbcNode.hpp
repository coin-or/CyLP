#ifndef ICbcNode_H
#define ICbcNode_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "ClpPackedMatrix.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "CoinFinite.hpp"
#include "CoinPragma.hpp"
#include "CbcNode.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
//#include "OsiClpSolverInterface.hpp"

class ICbcNode : public CbcNode{
public:
    bool breakTie(ICbcNode* y);
};


#endif
