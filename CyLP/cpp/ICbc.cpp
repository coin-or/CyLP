#include "ICbc.hpp"

ICbcModel* CbcSolveMIP(IClpSimplex* clpModel, PyObject* obj, 
        runTest_t runTest, runNewSolution_t runNewSolution,
        runEvery1000Nodes_t runEvery1000Nodes){
    OsiClpSolverInterface solver1(clpModel);
    solver1.initialSolve();
    ICbcModel*  model = new ICbcModel(solver1);
    CbcCompareUser compare(obj, 
            runTest, 
            runNewSolution, 
            runEvery1000Nodes);
    model->setNodeComparison(compare);
    model->branchAndBound();
    return model;
}

ICbcModel* CbcSolveMIP(IClpSimplex* clpModel){
    OsiClpSolverInterface solver1(clpModel);
    solver1.initialSolve();
    ICbcModel*  model = new ICbcModel(solver1);
    model->branchAndBound();
    return model;
}

