#include "IClpSimplex.hpp"

#include "ClpSimplexDual.hpp"
//#include "IClpSimplexPrimal.hpp"
#include "IClpSimplexPrimal.hpp"
#include "ClpSimplexPrimal.hpp"
#include "IClpPackedMatrix.hpp"
#include "OsiClpSolverInterface.hpp"
#include <sstream>

// define PyInt_* macros for Python 3.x
#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif


int IClpSimplex::argWeightedMax(PyObject* arr, PyObject* arr_ind, PyObject* w, PyObject* w_ind){
    //_import_array();

    npy_intp w_ind_len = PyArray_DIM(w_ind, 0);
    if (w_ind_len == 0)
        return -1; //return PyArray_ArgMax(reinterpret_cast<PyArrayObject*>(arr));

    int wIsNum = false;
    int wholeArray = false;

    double w_num_val;
    if (PyInt_Check(w)){
        wIsNum = true;
        w_num_val = (double)PyInt_AsLong(w);
    }else if (PyFloat_Check(w)){
        wIsNum = true;
        w_num_val = PyFloat_AsDouble(w);
    }else if (!PyArray_Check(w)){
        PyErr_SetString(PyExc_ValueError,
                "weights should be a number or a numpy array.");
        return -1;
    }


    if (PyInt_Check(arr_ind) || PyFloat_Check(arr_ind)){
        wholeArray = true;
    }else if (!PyArray_Check(arr_ind)){
        PyErr_SetString(PyExc_ValueError,
                "arr_ind should be a number(meaning 1..len(arr) or a numpy array.");
        return -1;
    }
    if (!PyArray_Check(arr) || !PyArray_Check(w_ind)){
        PyErr_SetString(PyExc_ValueError,
                "arr and w_ind should be numpy arrays.");
        return -1;
    }


    PyObject* arr_it = PyArray_IterNew(arr);

    npy_intp arr_len = PyArray_DIM(arr, 0);

    if (arr_len == 0)
        return 0;


    double maxVal;// = *(double*)PyArray_ITER_DATA(arr_it);
    int maxInd;// = 0;
    double curVal;
    double curInd;
    //int w_ind_val;
    //double w_val;



    //consider 4 cases:
    //1- whole array, weight is a single number
    //2- whole array, weights array
    //3- arr_ind, weights number
    //4- arr_ind, weights array
    if (wholeArray){
        maxVal = *(double*)PyArray_ITER_DATA(arr_it);
        maxInd = 0;
        PyObject* w_ind_it= PyArray_IterNew(w_ind);
        int w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);

        if (wIsNum){
            if (w_ind_val == 0){
                maxVal *= w_num_val;
                PyArray_ITER_NEXT(w_ind_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
            }
            PyArray_ITER_NEXT(arr_it);

            for (int i = 1 ; i < arr_len ; i++){
                curVal = *(double*)PyArray_ITER_DATA(arr_it);
                if (w_ind_val == i){
                    curVal *= w_num_val;
                    PyArray_ITER_NEXT(w_ind_it);
                    w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                }
                if (curVal > maxVal){
                    maxVal = curVal;
                    maxInd = i;
                }

                PyArray_ITER_NEXT(arr_it);
            }
        }
        else{ //look in whole array, weights array
            npy_intp w_len = PyArray_DIM(w, 0);
            npy_intp w_ind_len = PyArray_DIM(w_ind, 0);
            if (w_ind_len != w_len){
                PyErr_SetString(PyExc_ValueError,
                        "If w is a numpy array, w_ind should be a numpy array of the same size.");
                return -1;
            }

            PyObject* w_it= PyArray_IterNew(w);
            double w_val = *(double*)PyArray_ITER_DATA(w_it);
            if (w_ind_val == 0){
                maxVal *= w_val; //*(double *)PyArray_GETITEM(w, PyArray_GETPTR1(w, 0));
                PyArray_ITER_NEXT(w_ind_it);
                PyArray_ITER_NEXT(w_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                w_val = *(double*)PyArray_ITER_DATA(w_it);
            }

            PyArray_ITER_NEXT(arr_it);

            for (int i = 1 ; i < arr_len ; i++){
                curVal = *(double*)PyArray_ITER_DATA(arr_it);
                if (w_ind_val == i){
                    curVal *= w_val;//*(double*)PyArray_GETITEM(w, PyArray_GETPTR1(w, i));
                    PyArray_ITER_NEXT(w_ind_it);
                    PyArray_ITER_NEXT(w_it);
                    w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                    w_val = *(double*)PyArray_ITER_DATA(w_it);
                }
                if (curVal > maxVal){
                    maxVal = curVal;
                    maxInd = i;
                }

                PyArray_ITER_NEXT(arr_it);
            }

        }
    }
    else{  //only indices specified in arr_ind

        npy_intp arr_ind_len = PyArray_DIM(arr_ind, 0);
        npy_intp arr_len = PyArray_DIM(arr, 0);

        if (arr_ind_len != arr_len){
            PyErr_SetString(PyExc_ValueError,
                    "If a_ind is a numpy array, arr should be a numpy array of the same size.");
            return -1;
        }

        PyObject* arr_ind_it = PyArray_IterNew(arr_ind);
        int arr_ind_val = *(int*)PyArray_ITER_DATA(arr_ind_it);
        PyObject* arr_it = PyArray_IterNew(arr);
        double arr_val = *(double*)PyArray_ITER_DATA(arr_it);


        maxVal = arr_val;
        maxInd = 0 ; //arr_ind_val;
        //std::cout << "maxVal = " << maxVal << "\n";
        //std::cout << "maxInd = " << maxInd << "\n";


        if (wIsNum){
            PyObject* w_ind_it = PyArray_IterNew(w_ind);
            int w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
            while (PyArray_ITER_NOTDONE(w_ind_it) && arr_ind_val > w_ind_val){
                PyArray_ITER_NEXT(w_ind_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
            }
            if (arr_ind_val == w_ind_val){
                maxVal *= w_num_val;
                PyArray_ITER_NEXT(w_ind_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
            }

            for (int i = 1 ; i < arr_ind_len ; i++){
                PyArray_ITER_NEXT(arr_ind_it);
                PyArray_ITER_NEXT(arr_it);

                arr_ind_val = *(int*)PyArray_ITER_DATA(arr_ind_it);
                arr_val = *(double*)PyArray_ITER_DATA(arr_it); //*(double*)PyArray_GETITEM(arr, PyArray_GETPTR1(arr, arr_ind_val));

                while (PyArray_ITER_NOTDONE(w_ind_it) && arr_ind_val > w_ind_val){
                    PyArray_ITER_NEXT(w_ind_it);
                    w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                }

                if (arr_ind_val == w_ind_val)
                    arr_val *= w_num_val;
                if (arr_val > maxVal){
                    maxVal = arr_val;
                    maxInd = i;
                }


            }
        }
        else{  //just elements specified in arr_ind, weight's an array

            npy_intp arr_ind_len = PyArray_DIM(arr_ind, 0);
            npy_intp arr_len = PyArray_DIM(arr, 0);

            if (arr_ind_len != arr_len){
                PyErr_SetString(PyExc_ValueError,
                        "If a_ind is a numpy array, arr should be a numpy array of the same size.");
                return -1;
            }


            npy_intp w_len = PyArray_DIM(w, 0);
            npy_intp w_ind_len = PyArray_DIM(w_ind, 0);
            if (w_ind_len != w_len){
                PyErr_SetString(PyExc_ValueError,
                        "If w is a numpy array, w_ind should be a numpy array of the same size.");
                return -1;
            }

            PyObject* w_ind_it = PyArray_IterNew(w_ind);
            int w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
            PyObject* w_it = PyArray_IterNew(w);
            double w_val = *(double*)PyArray_ITER_DATA(w_it);


            while (PyArray_ITER_NOTDONE(w_ind_it) && arr_ind_val > w_ind_val){
                PyArray_ITER_NEXT(w_ind_it);
                PyArray_ITER_NEXT(w_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                w_val = *(double*)PyArray_ITER_DATA(w_it);
            }

            if (arr_ind_val == w_ind_val){
                maxVal *= w_val;
                PyArray_ITER_NEXT(w_ind_it);
                PyArray_ITER_NEXT(w_it);
                w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                w_val = *(double*)PyArray_ITER_DATA(w_it);

            }

            for (int i = 1 ; i < arr_ind_len ; i++){
                PyArray_ITER_NEXT(arr_ind_it);
                PyArray_ITER_NEXT(arr_it);

                arr_ind_val = *(int*)PyArray_ITER_DATA(arr_ind_it);
                arr_val = *(double*)PyArray_ITER_DATA(arr_it); //*(double*)PyArray_GETITEM(arr, PyArray_GETPTR1(arr, arr_ind_val));

                while (PyArray_ITER_NOTDONE(w_ind_it) && arr_ind_val > w_ind_val){
                    PyArray_ITER_NEXT(w_ind_it);
                    PyArray_ITER_NEXT(w_it);
                    w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                    w_val = *(double*)PyArray_ITER_DATA(w_it);
                }

                if (arr_ind_val == w_ind_val){
                    arr_val *= w_val;
                    PyArray_ITER_NEXT(w_ind_it);
                    PyArray_ITER_NEXT(w_it);
                    w_ind_val = *(int*)PyArray_ITER_DATA(w_ind_it);
                    w_val = *(double*)PyArray_ITER_DATA(w_it);

                }
                if (arr_val > maxVal){
                    maxVal = arr_val;
                    maxInd = i;
                }


            }
        }
    }
    return maxInd;
}


int IClpSimplex::argWeightedMax(PyObject* arr, PyObject* whr, double weight){
    //_import_array();
    if (!PyArray_Check(arr) || !PyArray_Check(whr)){
        PyErr_SetString(PyExc_ValueError,
                "Arguments of argWeightedMax should be numpy arrays.");
        return -1;
    }
    PyObject* arr_it = PyArray_IterNew(arr);
    PyObject* whr_it = PyArray_IterNew(whr);

    npy_intp arr_len = PyArray_DIM(arr, 0);

    if (arr_len == 0)
        return 0;

    double maxVal = *(double*)PyArray_ITER_DATA(arr_it);
    int maxInd = 0;
    double curVal;
    double curInd;

    int curWhere = *(int*)PyArray_ITER_DATA(whr_it);
    if (curWhere == 0){
        maxVal *= weight;
        PyArray_ITER_NEXT(whr_it);
        curWhere = *(int*)PyArray_ITER_DATA(whr_it);
    }

    PyArray_ITER_NEXT(arr_it);

    for (int i = 1 ; i < arr_len ; i++){
        curVal = *(double*)PyArray_ITER_DATA(arr_it);
        if (curWhere == i){
            curVal *= weight;
            PyArray_ITER_NEXT(whr_it);
            curWhere = *(int*)PyArray_ITER_DATA(whr_it);
        }
        if (curVal > maxVal){
            maxVal = curVal;
            maxInd = i;
        }

        PyArray_ITER_NEXT(arr_it);
    }
    return maxInd;
}

bool IClpSimplex::varIsFree(int ind){
    return getStatus(ind) == ClpSimplex::isFree;
}

bool IClpSimplex::varBasic(int ind){
    return getStatus(ind) == ClpSimplex::basic;
}

bool IClpSimplex::varAtUpperBound(int ind){
    return getStatus(ind) == ClpSimplex::atUpperBound;
}

bool IClpSimplex::varAtLowerBound(int ind){
    return getStatus(ind) == ClpSimplex::atLowerBound;
}

bool IClpSimplex::varSuperBasic(int ind){
    return getStatus(ind) == ClpSimplex::superBasic;
}


bool IClpSimplex::varIsFixed(int ind){
    return getStatus(ind) == ClpSimplex::isFixed;
}
PyObject* IClpSimplex::getStatusArray(){

    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_UINT8, this->status_ );

    return Arr;
}


PyObject* IClpSimplex::getReducedCosts(){

    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->djRegion() );

    return Arr;
}

void IClpSimplex::setReducedCosts(double* rc){
    int dim = getNumCols() + getNumRows();
    for (int i = 0; i < dim; i++) {
        dj_[i] = rc[i];
    }
}


PyObject* IClpSimplex::getComplementarityList(){

    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, QP_ComplementarityList );

    return Arr;
}

PyObject* IClpSimplex::getPivotVariable(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, this->pivotVariable() );

    return Arr;
}


PyObject* IClpSimplex::getPrimalRowSolution(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->primalRowSolution() );

    return Arr;
}

PyObject* IClpSimplex::getPrimalColumnSolution(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE, this->primalColumnSolution() );

    return Arr;
}

PyObject* IClpSimplex::getPrimalColumnSolutionAll(){
    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE, this->primalColumnSolution() );
    return Arr;
}

PyObject* IClpSimplex::getSolutionRegion(){
    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE, this->solutionRegion() );
    return Arr;
}

PyObject* IClpSimplex::getCostRegion(){
    npy_intp dims = getNumCols() + getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData(1, &dims, PyArray_DOUBLE, this->costRegion() );
    return Arr;
}

PyObject* IClpSimplex::getDualRowSolution(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->dualRowSolution() );

    return Arr;
}

PyObject* IClpSimplex::getDualColumnSolution(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->dualColumnSolution() );

    return Arr;
}

PyObject* IClpSimplex::getObjective(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->objective() );
    return Arr;
}

PyObject* IClpSimplex::getRowLower(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->rowLower() );
    return Arr;
}

PyObject* IClpSimplex::getRowUpper(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->rowUpper() );
    return Arr;
}

PyObject* IClpSimplex::getUpper(){

    npy_intp dims = getNumRows() + getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->upperRegion() );
    return Arr;
}

PyObject* IClpSimplex::getLower(){

    npy_intp dims = getNumRows() + getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->lowerRegion() );
    return Arr;
}

PyObject* IClpSimplex::getColLower(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->columnLower() );
    return Arr;
}

PyObject* IClpSimplex::getColUpper(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->columnUpper() );
    return Arr;
}

PyObject* IClpSimplex::getColumnScale(){

    npy_intp dims = getNumCols();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE,  columnScale_);
    return Arr;
}

PyObject* IClpSimplex::getRowScale(){

    npy_intp dims = getNumRows();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, rowScale_ );
    return Arr;
}



PyObject* IClpSimplex::getIntegerInformation(){
    npy_intp dims = getNumCols();
    PyObject* Arr;
    if (this->integerInformation())
        Arr = PyArray_SimpleNewFromData(1, &dims, PyArray_INT8, this->integerInformation());
    else
        Arr = PyArray_ZEROS(1, &dims, PyArray_INT8, 0);
    return Arr;
}

std::vector<std::string> IClpSimplex::getVariableNames(){
    if (lengthNames_)
        return columnNames_;
    return std::vector<std::string> ();
}

void IClpSimplex::setVariableName(int varInd,  char* name){
    if (varInd >= getNumCols())
        return;
    if (lengthNames_ == 0){
        unsigned int maxLength=0;
        int iRow;

        rowNames_ = std::vector<std::string> ();
        columnNames_ = std::vector<std::string> ();
        rowNames_.reserve(numberRows_);
        for (iRow=0;iRow<numberRows_;iRow++) {
            std::stringstream ss;
            ss << "r-";
            ss << iRow;
            std::string rowName = ss.str();
            if (rowName.length() > maxLength)
                maxLength = rowName.length();
            rowNames_.push_back(rowName);
        }

        columnNames_.reserve(numberColumns_);
        int iColumn;
        for (iColumn=0;iColumn<numberColumns_;iColumn++) {
            std::stringstream ss;
            ss << "c-";
            ss << iColumn;
            std::string colName = ss.str();
            if (colName.length() > maxLength)
                maxLength = colName.length();
            columnNames_.push_back(colName);
        }
//
//        int iColumn;
//        columnNames_.reserve(numberColumns_);
//        for (iColumn=0;iColumn<numberColumns_;iColumn++) {
//            const char * name = m.columnName(iColumn);
//            maxLength = CoinMax(maxLength,static_cast<unsigned int> (strlen(name)));
//            columnNames_.push_back(name);
//        }
        lengthNames_=static_cast<int> (maxLength);
//    columnNames_.resize(getNumCols());

    //std::cout << columnNamesAsChar()[10] << "<$$$$$$$$$$$$$\n";
    }
    std::string st(name);
    columnNames_[varInd] = st;
}

void IClpSimplex::setConstraintName(int constInd,  char* name){
    if (constInd >= getNumRows())
        return;
    if (lengthNames_ == 0){
        unsigned int maxLength=0;
        int iRow;

        rowNames_ = std::vector<std::string> ();
        columnNames_ = std::vector<std::string> ();
        rowNames_.reserve(numberRows_);
        for (iRow=0;iRow<numberRows_;iRow++) {
            std::stringstream ss;
            ss << "r-";
            ss << iRow;
            std::string rowName = ss.str();
            if (rowName.length() > maxLength)
                maxLength = rowName.length();
            rowNames_.push_back(rowName);
        }

        columnNames_.reserve(numberColumns_);
        int iColumn;
        for (iColumn=0;iColumn<numberColumns_;iColumn++) {
            std::stringstream ss;
            ss << "c-";
            ss << iColumn;
            std::string colName = ss.str();
            if (colName.length() > maxLength)
                maxLength = colName.length();
            columnNames_.push_back(colName);
        }
        lengthNames_=static_cast<int> (maxLength);
    }
    std::string st(name);
    rowNames_[constInd] = st;
}

void IClpSimplex::createTempArray(){
    tempIntArray = new int[getNumCols() + getNumRows()];
    tempArrayExists = true;
}

IClpSimplex::IClpSimplex(PyObject *obj_arg, runIsPivotAcceptable_t runIsPivotAcceptable_arg,
                         varSelCriteria_t runVarSelCriteria ):ClpSimplex()
                        {

    _import_array();
    tempArrayExists = false;
    obj = obj_arg;
    runIsPivotAcceptable = runIsPivotAcceptable_arg;
    varSelCriteria = runVarSelCriteria;
    customPrimal = 0;
    createStatus();
    pinfo = ClpPresolve();

    tempRow = NULL;
    tempRow_vector = NULL;
    QP_BanList = NULL;
    QP_ComplementarityList = NULL;
}




void IClpSimplex::useCustomPrimal(int u)
{
    customPrimal = u;
}

int IClpSimplex::getUseCustomPrimal()
{
    return customPrimal;
}

void IClpSimplex::setComplementarityList(int * cl)
{
    QP_ComplementarityList = cl;
}

int* IClpSimplex::ComplementarityList()
{
    return QP_ComplementarityList;
}

void IClpSimplex::setBasisStatus(const int* cstat, const int* rstat){
    OsiClpSolverInterface osi(this, false);
    osi.setBasisStatus(cstat, rstat);
    return;
}

void IClpSimplex::setMaxNumIteration(int m){
    setIntParam(ClpMaxNumIteration, m);
}

void IClpSimplex::getBasisStatus(int* cstat, int* rstat){
    OsiClpSolverInterface osi(this, false);
    osi.getBasisStatus(cstat, rstat);
    return;
}

IClpSimplex::IClpSimplex (ClpSimplex * wholeModel,
        int numberColumns, const int * whichColumns):
    ClpSimplex(wholeModel, numberColumns, whichColumns)
{
    _import_array();
    tempArrayExists = false;
    tempRow_vector = NULL;
    tempRow = NULL;

    QP_ComplementarityList = NULL;
    QP_BanList = NULL;//new int[nvars];
    pinfo = ClpPresolve();
}


IClpSimplex::~IClpSimplex(){
    if (QP_ComplementarityList)
        delete QP_ComplementarityList;

    if (QP_BanList)
        delete QP_BanList;

    if (tempRow)
        delete tempRow;

    if (tempRow_vector)
        delete tempRow_vector;

}


void IClpSimplex::dualExpanded(ClpSimplex * model,CoinIndexedVector * array,
        double * other,int mode){

    this->clpMatrix()->dualExpanded(model,array,other,mode);
}

int IClpSimplex::isPivotAcceptable()
{
    if (this->obj && this->runIsPivotAcceptable) {
        return this->runIsPivotAcceptable(this->obj);
    }
    std::cerr << "** pivotRow: invalid cy-state: obj [" << this->obj << "] fct: ["
        << this->runIsPivotAcceptable << "]\n";
    return -1;
}

void IClpSimplex::setCriteria(varSelCriteria_t vsc){
    varSelCriteria = vsc;
}

int IClpSimplex::checkVar(int varInd){
    if (this->obj && this->varSelCriteria) {
        return this->varSelCriteria(this->obj, varInd);
    }
    std::cerr << "** pivotRow: invalid cy-state: obj [" << this->obj << "] fct: ["
        << this->varSelCriteria << "]\n";
    return -1;

}


ICbcModel* IClpSimplex::getICbcModel(){
    // ?
    matrix_->setDimensions(numberRows_, numberColumns_);

    OsiClpSolverInterface solver1(this);
    ICbcModel*  model = new ICbcModel(solver1);
    return model;
}

void  IClpSimplex::writeLp(const char *filename,
                       const char *extension,
                       double epsilon,
                       int numberAcross,
                       int decimals,
                       double objSense,
                       bool useRowNames)
    {
    matrix_->setDimensions(numberRows_, numberColumns_);

    OsiClpSolverInterface solver1(this);
    solver1.writeLp(filename, extension, epsilon, numberAcross, decimals, objSense, useRowNames);
    return ;

    }



//Get a column of the tableau
    void
IClpSimplex::getBInvACol(int col, double* vec)
{
    if (!rowArray_[0]) {
        printf("ClpSimplexPrimal or ClpSimplexDual should have been called with correct startFinishOption\n");
        abort();
    }
    CoinIndexedVector * rowArray0 = rowArray(0);
    CoinIndexedVector * rowArray1 = rowArray(1);
    rowArray0->clear();
    rowArray1->clear();
    // get column of matrix
#ifndef NDEBUG
    int n = numberColumns_+numberRows_;
    if (col<0||col>=n) {
        //indexError(col,"getBInvACol");
    }
#endif
    if (!rowScale_) {
        if (col<numberColumns_) {
            unpack(rowArray1,col);
        } else {
            rowArray1->insert(col-numberColumns_,1.0);
        }
    } else {
        if (col<numberColumns_) {
            unpack(rowArray1,col);
            double multiplier = 1.0*inverseColumnScale_[col];
            int number = rowArray1->getNumElements();
            int * index = rowArray1->getIndices();
            double * array = rowArray1->denseVector();
            for (int i=0;i<number;i++) {
                int iRow = index[i];
                // make sure not packed
                assert (array[iRow]);
                array[iRow] *= multiplier;
            }
        } else {
            rowArray1->insert(col-numberColumns_,rowScale_[col-numberColumns_]);
        }
    }
    factorization_->updateColumn(rowArray0,rowArray1,false);
    // But swap if pivot variable was slack as clp stores slack as -1.0
    double * array = rowArray1->denseVector();
    if (!rowScale_) {
        for (int i=0;i<numberRows_;i++) {
            double multiplier = (pivotVariable_[i]<numberColumns_) ? 1.0 : -1.0;
            vec[i] = multiplier * array[i];
        }
    } else {
        for (int i=0;i<numberRows_;i++) {
            int pivot = pivotVariable_[i];
            if (pivot<numberColumns_)
                vec[i] = array[i] * columnScale_[pivot];
            else
                vec[i] = - array[i] / rowScale_[pivot-numberColumns_];
        }
    }
    rowArray1->clear();
}


//Fetches the ncol th column into colArray
void IClpSimplex::getACol(int ncol, CoinIndexedVector * colArray){


    //CoinIndexedVector * colArray = temp_rowArray[1];

    colArray->clear();

    // get column of matrix
#ifndef NDEBUG
    //int n = numberColumns_+numberRows_;
    //if (ncol<0||ncol>=n) {
    //	indexError(ncol,"getBInvACol");
    //}
#endif

    if (!rowScale()) {
        if (ncol<numberColumns()) {
            unpack(colArray,ncol);
        } else {
            colArray->insert(ncol- numberColumns(),1.0);
        }
    } else {
        if (ncol<numberColumns()) {
            unpack(colArray,ncol);
            double multiplier = 1.0* inverseColumnScale()[ncol];
            int number = colArray->getNumElements();
            int * index = colArray->getIndices();
            double * array = colArray->denseVector();
            for (int i=0;i<number;i++) {
                int iRow = index[i];
                // make sure not packed
                assert (array[iRow]);
                array[iRow] *= multiplier;
            }
        } else {
            colArray->insert(ncol- numberColumns(),rowScale()[ncol-numberColumns()]);
        }
    }

}


void IClpSimplex::getRightHandSide(double* righthandside)
{
    int nr=numberRows();

    extractSenseRhsRange(righthandside);

    int* basis_index = pivotVariable();


    //FIXME: change these lines to be like getColSoution and getRowActivity in OsiClp
    const double *solution = solutionRegion(1);
    const double *row_act = solutionRegion(0);

    //FIXME: This must be fixed. The first line causes seg fault
    //So I'm allocating and deleting in this function
    //double *slack_val = tempRow;
    double *slack_val = new double[nr];

    for(int i=0; i<nr; i++) {
        slack_val[i] = righthandside[i] - row_act[i];
    }

    int ncol = numberColumns();
    for (int i = 0 ; i < nr; i++) {
        if (basis_index[i] < ncol){
            righthandside[i] = solution[basis_index[i]];
            //std::cout << "sim: rhs " << i << " = " << solution[basis_index[i]] << "\n";
        }else {
            righthandside[i] = slack_val[basis_index[i]-ncol];
            //std::cout << "sim: rhs " << i << " = " << slack_val[basis_index[i]-ncol] << "\n";
        }

    }

    delete slack_val;

    //return righthandside;
}


void IClpSimplex::extractSenseRhsRange(double* rhs_)
{
    //if (rowsense_ == NULL) {
    // all three must be NULL
    //assert ((rhs_ == NULL) && (rowrange_ == NULL));

    int nr=numberRows();
    if ( nr!=0 ) {
        //char* rowsense_ = new char[nr];
        //double* rhs_ = new double[nr];
        //double* rowrange_ = PE_tempRow[0]; //new double[nr];
        //std::fill(rowrange_,rowrange_+nr,0.0);

        const double * lb = rowLower();
        const double * ub = rowUpper();

        int i;
        for ( i=0; i<nr; i++ ) {
            //std::cout << i << " : " << lb[i] << ", " << ub[i] << "\n";
            //convertBoundToSense(lb[i], ub[i], rowsense_[i], rhs_[i], rowrange_[i]);
            convertBoundToSense(lb[i], ub[i], rhs_[i]);
        }

        //delete rowrange_;
        //delete rowsense_;
    }

}

void IClpSimplex::convertBoundToSense(const double lower, const double upper,
        double& right)
{
   double inf = pow(double(10), 16);
    if (lower > -inf) {
        if (upper < inf) {
            right = upper;
            //if (upper==lower) {
            //sense = 'E';
            //} else {
            //sense = 'R';
            //range = upper - lower;
            //}
        } else {
            //sense = 'G';
            right = lower;
        }
    } else {
        if (upper < inf) {
            //sense = 'L';
            right = upper;
        } else {
            //sense = 'N';
            right = 0.0;
        }
    }
}


void IClpSimplex::vectorTimesB_1(CoinIndexedVector* vec){
    factorization_->updateColumnTranspose(tempRow_vector, vec);
}



void IClpSimplex::transposeTimesSubset(int number, int* which, double* pi, double* y){
    reinterpret_cast<IClpPackedMatrix*>(matrix_)->transposeTimesSubset(number,
                             which, pi, y, rowScale(), columnScale(), NULL);
}


void IClpSimplex::transposeTimes(const ClpSimplex * model, double scalar,
                                 const CoinIndexedVector * x,
                                 CoinIndexedVector * y,
                                 CoinIndexedVector * z){
    reinterpret_cast<IClpPackedMatrix*>(matrix_)->transposeTimes(
                                model, scalar, x, y, z);
}


void IClpSimplex::transposeTimesSubsetAll(int number, long long int* which, double* pi, double* y){
    reinterpret_cast<IClpPackedMatrix*>(matrix_)->transposeTimesSubsetAll(this, number,
                             which, pi, y, rowScale(), columnScale(), NULL);
}

// Copy constructor.
IClpSimplex::IClpSimplex(const ClpSimplex &rhs,PyObject *obj,
                            runIsPivotAcceptable_t runIsPivotAcceptable,
                            varSelCriteria_t varSelCriteria,
                            int useCustomPrimal, int scalingMode ) :
  ClpSimplex(rhs,scalingMode),
    obj(obj),
    runIsPivotAcceptable(runIsPivotAcceptable),
    varSelCriteria(varSelCriteria),
    customPrimal(useCustomPrimal),
    tempArrayExists(false),
    tempRow(NULL),
    tempRow_vector(NULL),
    QP_BanList(NULL),
    QP_ComplementarityList(NULL)
{
}




IClpSimplex* IClpSimplex::preSolve(IClpSimplex* si,
        double feasibilityTolerance,
        bool keepIntegers,
        int numberPasses,
        bool dropNames,
        bool doRowObjective)
{
    //ClpPresolve pinfo;
    ClpSimplex* s = pinfo.presolvedModel(*si,feasibilityTolerance,
                            keepIntegers, numberPasses, dropNames, doRowObjective);
    if (s)
        {
        IClpSimplex* ret = new IClpSimplex(*s, si->obj, si->runIsPivotAcceptable, si->varSelCriteria, si->customPrimal);
        //ret->pinfo = pinfo;
        //pinfo.postsolve();
        return ret;
        }

    return NULL;
}

void IClpSimplex::postSolve(bool updateStatus){
    pinfo.postsolve(updateStatus);
}


int IClpSimplex::dualWithPresolve(IClpSimplex* si,
        double feasibilityTolerance,
        bool keepIntegers,
        int numberPasses,
        bool dropNames,
        bool doRowObjective)
{
    ClpPresolve pinfoTemp;
    ClpSimplex* s = pinfoTemp.presolvedModel(*si,feasibilityTolerance,
                            keepIntegers, numberPasses, dropNames, doRowObjective);
    if (s)
        {
        int ret = s->dual();
        pinfoTemp.postsolve();
        delete s;
        checkSolution();
        dual();
        return ret;
        }
    return -2000;
}


int IClpSimplex::primalWithPresolve(IClpSimplex* si,
        double feasibilityTolerance,
        bool keepIntegers,
        int numberPasses,
        bool dropNames,
        bool doRowObjective)
{
    ClpPresolve pinfoTemp;
    ClpSimplex* s = pinfoTemp.presolvedModel(*si,feasibilityTolerance,
                            keepIntegers, numberPasses, dropNames, doRowObjective);
    if (s)
        {
        int ret = s->primal();
        pinfoTemp.postsolve();
        delete s;
        checkSolution();
        primal();
        return ret;
        }
    return -2000;
}

int IClpSimplex::initialSolve(int preSolveType){
    ClpSolve options;
    options.setPresolveType(static_cast<ClpSolve::PresolveType>(preSolveType));
    return ClpSimplex::initialSolve(options);
}


int IClpSimplex::primal (int ifValuesPass , int startFinishOptions)
{
    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if (tempRow == NULL)
        tempRow = new double[numberRows()];
    if (tempRow_vector == NULL)
        tempRow_vector = new CoinIndexedVector();
    if (QP_BanList == NULL)
        QP_BanList = new int[numberColumns() + numberRows()];
    //FIXME: This is a crazy 1000 here.
    //But whatever you do to fix this try it on adlittle, degen2
    tempRow_vector->reserve(numberRows() + numberColumns() + numberExtraRows() + 1000);
    //tempRow_vector->reserve(numberRows() + +numberColumns() + numberExtraRows());


    //double savedPivotTolerance = factorization_->pivotTolerance();
#ifndef SLIM_CLP
    // See if nonlinear
    if (objective_->type()>1&&objective_->activated())
        return reducedGradient();
#endif
    CoinAssert ((ifValuesPass>=0&&ifValuesPass<3)||
            (ifValuesPass>=12&&ifValuesPass<100)||
            (ifValuesPass>=112&&ifValuesPass<200));
    if (ifValuesPass>=12) {
        int numberProblems = (ifValuesPass-10)%100;
        ifValuesPass = (ifValuesPass<100) ? 1 : 2;
        // Go parallel to do solve
        // Only if all slack basis
        int i;
        for ( i=0;i<numberColumns_;i++) {
            if (getColumnStatus(i)==basic)
                break;
        }
        if (i==numberColumns_) {
            // try if vaguely feasible
            CoinZeroN(ClpModel::rowActivity_,numberRows_);
            const int * row = matrix_->getIndices();
            const CoinBigIndex * columnStart = matrix_->getVectorStarts();
            const int * columnLength = matrix_->getVectorLengths();
            const double * element = matrix_->getElements();
            for (int iColumn=0;iColumn<numberColumns_;iColumn++) {
                CoinBigIndex j;
                double value = ClpModel::columnActivity_[iColumn];
                if (value) {
                    CoinBigIndex start = columnStart[iColumn];
                    CoinBigIndex end = start + columnLength[iColumn];
                    for (j=start; j<end; j++) {
                        int iRow=row[j];
                        ClpModel::rowActivity_[iRow] += value*element[j];
                    }
                }
            }
            checkSolutionInternal();
            if (sumPrimalInfeasibilities_*sqrt(static_cast<double>(numberRows_))<1.0) {
                // Could do better if can decompose
                // correction to get feasible
                double scaleFactor = 1.0/numberProblems;
                double * correction = new double [numberRows_];
                for (int iRow=0;iRow<numberRows_;iRow++) {
                    double value=ClpModel::rowActivity_[iRow];
                    if (value>rowUpper_[iRow])
                        value = rowUpper_[iRow]-value;
                    else if (value<rowLower_[iRow])
                        value = rowLower_[iRow]-value;
                    else
                        value=0.0;
                    correction[iRow]=value*scaleFactor;
                }
                int numberColumns = (numberColumns_+numberProblems-1)/numberProblems;
                int * whichRows = new int [numberRows_];
                for (int i=0;i<numberRows_;i++)
                    whichRows[i]=i;
                int * whichColumns = new int [numberColumns_];
                ClpSimplex ** model = new ClpSimplex * [numberProblems];
                int startColumn=0;
                double * saveLower = CoinCopyOfArray(rowLower_,numberRows_);
                double * saveUpper = CoinCopyOfArray(rowUpper_,numberRows_);
                for (int i=0;i<numberProblems;i++) {
                    int endColumn = CoinMin(startColumn+numberColumns,numberColumns_);
                    CoinZeroN(ClpModel::rowActivity_,numberRows_);
                    for (int iColumn=startColumn;iColumn<endColumn;iColumn++) {
                        whichColumns[iColumn-startColumn]=iColumn;
                        CoinBigIndex j;
                        double value = ClpModel::columnActivity_[iColumn];
                        if (value) {
                            CoinBigIndex start = columnStart[iColumn];
                            CoinBigIndex end = start + columnLength[iColumn];
                            for (j=start; j<end; j++) {
                                int iRow=row[j];
                                ClpModel::rowActivity_[iRow] += value*element[j];
                            }
                        }
                    }
                    // adjust rhs
                    for (int iRow=0;iRow<numberRows_;iRow++) {
                        double value=ClpModel::rowActivity_[iRow]+correction[iRow];
                        if (saveUpper[iRow]<1.0e30)
                            rowUpper_[iRow]=value;
                        if (saveLower[iRow]>-1.0e30)
                            rowLower_[iRow]=value;
                    }
                    model[i] = new ClpSimplex(this,numberRows_,whichRows,
                            endColumn-startColumn,whichColumns);
                    //#define FEB_TRY
#ifdef FEB_TRY
                    model[i]->setPerturbation(perturbation_);
#endif
                    startColumn=endColumn;
                }
                memcpy(rowLower_,saveLower,numberRows_*sizeof(double));
                memcpy(rowUpper_,saveUpper,numberRows_*sizeof(double));
                delete [] saveLower;
                delete [] saveUpper;
                delete [] correction;
                // solve (in parallel)
                for (int i=0;i<numberProblems;i++) {
                    model[i]->primal(1/*ifValuesPass*/);
                }
                startColumn=0;
                int numberBasic=0;
                // use whichRows as counter
                for (int iRow=0;iRow<numberRows_;iRow++) {
                    int startValue=0;
                    if (rowUpper_[iRow]>rowLower_[iRow])
                        startValue++;
                    if (rowUpper_[iRow]>1.0e30)
                        startValue++;
                    if (rowLower_[iRow]<-1.0e30)
                        startValue++;
                    whichRows[iRow]=1000*startValue;
                }
                for (int i=0;i<numberProblems;i++) {
                    int endColumn = CoinMin(startColumn+numberColumns,numberColumns_);
                    ClpSimplex * simplex = model[i];
                    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    // this line changed from columnActivity_ to getColSolution because columnactivity in protected in ClpModel
                    const double * solution = getColSolution();
                    for (int iColumn=startColumn;iColumn<endColumn;iColumn++) {
                        ClpModel::columnActivity_[iColumn] = solution[iColumn-startColumn];
                        Status status = simplex->getColumnStatus(iColumn-startColumn);
                        setColumnStatus(iColumn,status);
                        if (status==basic)
                            numberBasic++;
                    }
                    for (int iRow=0;iRow<numberRows_;iRow++) {
                        if (simplex->getRowStatus(iRow)==basic)
                            whichRows[iRow]++;
                    }
                    delete model[i];
                    startColumn=endColumn;
                }
                delete [] model;
                for (int iRow=0;iRow<numberRows_;iRow++)
                    setRowStatus(iRow,superBasic);
                CoinZeroN(ClpModel::rowActivity_,numberRows_);
                for (int iColumn=0;iColumn<numberColumns_;iColumn++) {
                    CoinBigIndex j;
                    double value = ClpModel::columnActivity_[iColumn];
                    if (value) {
                        CoinBigIndex start = columnStart[iColumn];
                        CoinBigIndex end = start + columnLength[iColumn];
                        for (j=start; j<end; j++) {
                            int iRow=row[j];
                            ClpModel::rowActivity_[iRow] += value*element[j];
                        }
                    }
                }
                checkSolutionInternal();
                if (numberBasic<numberRows_) {
                    int * order = new int [numberRows_];
                    for (int iRow=0;iRow<numberRows_;iRow++) {
                        setRowStatus(iRow,superBasic);
                        int nTimes = whichRows[iRow]%1000;
                        if (nTimes)
                            nTimes += whichRows[iRow]/500;
                        whichRows[iRow]=-nTimes;
                        order[iRow]=iRow;
                    }
                    CoinSort_2(whichRows,whichRows+numberRows_,order);
                    int nPut = numberRows_-numberBasic;
                    for (int i=0;i<nPut;i++) {
                        int iRow = order[i];
                        setRowStatus(iRow,basic);
                    }
                    delete [] order;
                } else if (numberBasic>numberRows_) {
                    double * away = new double [numberBasic];
                    numberBasic=0;
                    for (int iColumn=0;iColumn<numberColumns_;iColumn++) {
                        if (getColumnStatus(iColumn)==basic) {
                            double value = ClpModel::columnActivity_[iColumn];
                            value = CoinMin(value-columnLower_[iColumn],
                                    columnUpper_[iColumn]-value);
                            away[numberBasic]=value;
                            whichColumns[numberBasic++]=iColumn;
                        }
                    }
                    CoinSort_2(away,away+numberBasic,whichColumns);
                    int nPut = numberBasic-numberRows_;
                    for (int i=0;i<nPut;i++) {
                        int iColumn = whichColumns[i];
                        double value = ClpModel::columnActivity_[iColumn];
                        if (value-columnLower_[iColumn]<
                                columnUpper_[iColumn]-value)
                            setColumnStatus(iColumn,atLowerBound);
                        else
                            setColumnStatus(iColumn,atUpperBound);
                    }
                    delete [] away;
                }
                delete [] whichColumns;
                delete [] whichRows;
            }
        }
    }
    /*  Note use of "down casting".  The only class the user sees is ClpSimplex.
        Classes ClpSimplexDual, ClpSimplexPrimal, (ClpSimplexNonlinear)
        and ClpSimplexOther all exist and inherit from ClpSimplex but have no
        additional data and have no destructor or (non-default) constructor.

        This is to stop classes becoming too unwieldy and so I (JJF) can use e.g. "perturb"
        in primal and dual.

        As far as I can see this is perfectly safe.
        */

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    int returnCode;
    if (!customPrimal){
        //std::cout << "IClpSimplex: continue with ClpSimplexPrimal   \n";
        returnCode = reinterpret_cast<ClpSimplexPrimal *> (this)->primal(ifValuesPass,startFinishOptions);
    }
    else {
        //std::cout << "IClpSimplex L280: casting to IClpSimplexPrimal\n";
        returnCode = reinterpret_cast<IClpSimplexPrimal *> (this)->primal(ifValuesPass,startFinishOptions);
    }
    //int lastAlgorithm=1;

    if (problemStatus_==10) {
        //lastAlgorithm=-1;
        //printf("Cleaning up with dual\n");
        int savePerturbation = perturbation_;
        perturbation_=100;
        bool denseFactorization = initialDenseFactorization();
        // It will be safe to allow dense
        setInitialDenseFactorization(true);
        // check which algorithms allowed
        int dummy;
        baseIteration_=numberIterations_;
        if ((matrix_->generalExpanded(this,4,dummy)&2)!=0&&(specialOptions_&8192)==0) {
            double saveBound = dualBound_;
            // upperOut_ has largest away from bound
            dualBound_=CoinMin(CoinMax(2.0*upperOut_,1.0e8),dualBound_);
            returnCode = reinterpret_cast<ClpSimplexDual *> (this)->dual(0,startFinishOptions);
            dualBound_=saveBound;
        } else {
            if (!customPrimal){
                std::cout << "IClpSimplex: continue with ClpSimplexPrimal   \n";
                returnCode = reinterpret_cast<ClpSimplexPrimal *> (this)->primal(0,startFinishOptions);
            }else {
                std::cout << "IClpSimplex: casting to IClpSimplexPrimal\n";
                returnCode = reinterpret_cast<IClpSimplexPrimal *> (this)->primal(0,startFinishOptions);
            }
        }

        baseIteration_=0;
        setInitialDenseFactorization(denseFactorization);
        perturbation_=savePerturbation;
        if (problemStatus_==10)
            problemStatus_=0;
    }
    //factorization_->pivotTolerance(savedPivotTolerance);
    onStopped(); // set secondary status if stopped
    //if (problemStatus_==1&&lastAlgorithm==1)
    //returnCode=10; // so will do primal after postsolve

    //delete tempRow;
    //delete tempRow_vector;
    //delete QP_BanList;


    return returnCode;
}


double cdot(CoinIndexedVector* pv1, CoinIndexedVector* pv2){
    double sum = 0;
    int	size = pv1->getNumElements();
    int* indices = pv1->getIndices();

    for (int i = 0; i < size; i++)
        sum += (*pv1)[indices[i]] * (*pv2)[indices[i]];
    return sum;

}

PyObject* IClpSimplex::filterVars(PyObject* inds){
    if (!PyArray_Check(inds)){
        PyErr_SetString(PyExc_ValueError,
                "filterVars: inds should be a numpy array.");
        return NULL;
    }

    npy_intp inds_len = PyArray_DIM(inds, 0);
    if (inds_len == 0){
        npy_intp dims = 0;
        PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT, tempIntArray );
        return Arr;
    }
    if (tempArrayExists == false)
        createTempArray();

    int ind_count = 0;

    PyObject* inds_it= PyArray_IterNew(inds);
    int i;
    double bestRc = 0;
    double* rc = djRegion();
    while (PyArray_ITER_NOTDONE(inds_it)){
        i = *(int*)PyArray_ITER_DATA(inds_it);
        if (fabs(rc[i]) < bestRc){
            PyArray_ITER_NEXT(inds_it);
            continue;
        }
        if (checkVar(i)){
            tempIntArray[ind_count++] = i;
            bestRc = fabs(rc[i]);
        }
        PyArray_ITER_NEXT(inds_it);
    }

    npy_intp dims = ind_count;
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT, tempIntArray );
    return Arr;

}

