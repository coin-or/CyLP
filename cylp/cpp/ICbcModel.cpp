#include "ICbcModel.hpp"

#include "CbcCompareUser.hpp"
#include "CbcSolver.hpp"

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


int ICbcModel::cbcMain(){
        // initialize
        int returnCode = -1;
	int logLevel = this->logLevel();
        CbcMain0(*this);
        const char* argv[] = {"ICbcModel", "-solve","-quit"};
        // Solve ICbcModel and quit
	this->setLogLevel(logLevel);
        returnCode = CbcMain1 (3, argv, *this);
        return returnCode;
}
