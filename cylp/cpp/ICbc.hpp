//#define NPY_NO_DEPRECATED_API

#include "CbcConfig.h"

// For Branch and bound
#include "OsiSolverInterface.hpp"
#include "ICbcModel.hpp"

#include "OsiClpSolverInterface.hpp"
#include "ClpPresolve.hpp"
//#include "CbcCompareUser.hpp"
#include "CglProbing.hpp"
#include "IClpSimplex.hpp"
#include "CbcCompareUser.hpp"

ICbcModel* CbcSolveMIP(IClpSimplex* model,
                       PyObject* obj,
                       runTest_t runTest,
                       runNewSolution_t runNewSolution,
                       runEvery1000Nodes_t runEvery1000Nodes);

ICbcModel* CbcSolveMIP(IClpSimplex* model);

