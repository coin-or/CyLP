#ifndef ICbcModel_H
#define ICbcModel_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "ClpPackedMatrix.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "CoinFinite.hpp"
#include "CoinPragma.hpp"
#include "CbcModel.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "OsiClpSolverInterface.hpp"
#include "ICbcNode.hpp"
//#include "CbcSolver.hpp"
//#include "CbcCompareUser.hpp"

class ICbcModel;
typedef int (*runTest_t)(void *instance, ICbcNode * x, ICbcNode * y);
typedef bool (*runNewSolution_t)(void *instance,ICbcModel * model,
                       double objectiveAtContinuous,
                       int numberInfeasibilitiesAtContinuous);
typedef int (*runEvery1000Nodes_t)(void *instance,
                            ICbcModel * model,int numberNodes);



class ICbcModel : public CbcModel{
public:
    ICbcModel(OsiClpSolverInterface&);
    PyObject * getPrimalVariableSolution();

    void setNodeCompare(PyObject* obj,
                           runTest_t runTest, runNewSolution_t runNewSolution,
                           runEvery1000Nodes_t runEvery1000Nodes);
    int cbcMain();
};


#endif
