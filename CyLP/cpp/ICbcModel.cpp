#include "ICbcModel.hpp"

#include "CbcCompareUser.hpp"

PyObject* ICbcModel::getPrimalVariableSolution(){

    _import_array();
    npy_intp dims = this->solver()->getNumCols();
    double* d = (double*)(this->solver()->getColSolution()); 
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, d );
    
    return Arr;
}

ICbcModel::ICbcModel(OsiClpSolverInterface& osiint):CbcModel(osiint){
    _import_array();
}

void ICbcModel::setNodeCompare(PyObject* obj,
                           runTest_t runTest, runNewSolution_t runNewSolution,
                           runEvery1000Nodes_t runEvery1000Nodes){
    CbcCompareUser compare(obj, runTest,
                           runNewSolution,runEvery1000Nodes);
    setNodeComparison(compare);

}


