#include "ICoinMpsIO.hpp"

double * ICoinMpsIO::IRowLower() const
{
  return rowlower_;
}
double * ICoinMpsIO::IRowUpper() const
{
  return rowupper_;
}

double * ICoinMpsIO::IColLower() const
{
  return collower_;
}
double * ICoinMpsIO::IColUpper() const
{
  return colupper_;
}


char * ICoinMpsIO::IRowSense() const
{
  if ( rowsense_==NULL ) {

    int nr=numberRows_;
    rowsense_ = reinterpret_cast<char *> (malloc(nr*sizeof(char)));


    double dum1,dum2;
    int i;
    for ( i=0; i<nr; i++ ) {
      convertBoundToSense(rowlower_[i],rowupper_[i],rowsense_[i],dum1,dum2);
    }
  }
  return rowsense_;
}



double * ICoinMpsIO::IRightHandSide() const
{
  if ( rhs_==NULL ) {

    int nr=numberRows_;
    rhs_ = reinterpret_cast<double *> (malloc(nr*sizeof(double)));


    char dum1;
    double dum2;
    int i;
    for ( i=0; i<nr; i++ ) {
      convertBoundToSense(rowlower_[i],rowupper_[i],dum1,rhs_[i],dum2);
    }
  }
  return rhs_;
}


double * ICoinMpsIO::IRowRange() const
{
  if ( rowrange_==NULL ) {

    int nr=numberRows_;
    rowrange_ = reinterpret_cast<double *> (malloc(nr*sizeof(double)));
    std::fill(rowrange_,rowrange_+nr,0.0);

    char dum1;
    double dum2;
    int i;
    for ( i=0; i<nr; i++ ) {
      convertBoundToSense(rowlower_[i],rowupper_[i],dum1,dum2,rowrange_[i]);
    }
  }
  return rowrange_;
}

char * ICoinMpsIO::IintegerColumns() const
{
  return integerType_;
}

double * ICoinMpsIO::IObjCoefficients() const
{
  return objective_;
}

PyObject* ICoinMpsIO::np_getColLower(){
	npy_intp dims = this->getNumCols();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IColLower() );
	return Arr;
}


PyObject* ICoinMpsIO::np_getColUpper(){
	npy_intp dims = this->getNumCols();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IColUpper() );
	return Arr;
}

PyObject* ICoinMpsIO::np_getRowSense(){
	npy_intp dims = this->getNumRows();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT8, this->IRowSense() );
	return Arr;
}


PyObject* ICoinMpsIO::np_getRightHandSide(){
	npy_intp dims = this->getNumRows();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IRightHandSide() );
	return Arr;
}

PyObject* ICoinMpsIO::np_getRowRange(){
	npy_intp dims = this->getNumRows();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IRowRange() );
	return Arr;
}

PyObject* ICoinMpsIO::np_getRowLower(){
	npy_intp dims = this->getNumRows();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IRowLower() );
	return Arr;
}

PyObject* ICoinMpsIO::np_getRowUpper(){
	npy_intp dims = this->getNumRows();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IRowUpper() );
	return Arr;
}

PyObject* ICoinMpsIO::np_getObjCoefficients(){
	npy_intp dims = this->getNumCols();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IObjCoefficients() );
	return Arr;
}

PyObject* ICoinMpsIO::np_integerColumns(){
	npy_intp dims = this->getNumCols();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT8, this->IintegerColumns() );
	return Arr;
}


PyObject* ICoinMpsIO::getQPColumnStarts(){
	npy_intp dims = this->getNumCols() + 1;
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, d_colStart );
	return Arr;
}

PyObject* ICoinMpsIO::getQPColumns(){
	npy_intp dims = d_colStart[this->getNumCols()];
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, d_cols );
	return Arr;
}

PyObject* ICoinMpsIO::getQPElements(){
	npy_intp dims = d_colStart[this->getNumCols()];
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, d_elements );
	return Arr;
}


double ICoinMpsIO::getObjectiveOffset(){
	return objectiveOffset_;
}

int ICoinMpsIO::IreadQuadraticMps(const char * filename, int checkSymmetry){



	int ret = readQuadraticMps(NULL, d_colStart, d_cols, d_elements, checkSymmetry);

	return ret;
}

ICoinMpsIO::ICoinMpsIO():CoinMpsIO()
{
	d_colStart = NULL;
	d_cols = NULL;
	d_elements = NULL;
}

ICoinMpsIO::~ICoinMpsIO(){
// 	if (d_colStart)
// 		delete[] d_colStart;
// 	if (d_cols)
// 		delete[] d_cols;
// 	if (d_elements)
// 		delete[] d_elements;
}


//------------------------------------------------------------------
// Create a row copy of the matrix ...
//------------------------------------------------------------------
ICoinPackedMatrix * ICoinMpsIO::IgetMatrixByRow() const
{
  if ( matrixByRow_ == NULL && matrixByColumn_) {
    matrixByRow_ = new CoinPackedMatrix(*matrixByColumn_);
    matrixByRow_->reverseOrdering();
  }
  return static_cast<ICoinPackedMatrix*>(matrixByRow_);
}

//------------------------------------------------------------------
// Create a column copy of the matrix ...
//------------------------------------------------------------------
ICoinPackedMatrix * ICoinMpsIO::IgetMatrixByCol() const
{
  return static_cast<ICoinPackedMatrix*>(matrixByColumn_);
}


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
/* The Following part is exactly as defined in CoinMpsIO and is added
because otherwise we will get this error:

Traceback (most recent call last):
  File "p.py", line 10, in <module>
    from CyCoinMpsIO import CyCoinMpsIO
ImportError: dlopen(/Users/mehdi/Documents/qp/codes/CyClp/CyCoinMpsIO.so, 2): Symbol not found: __ZNK10ICoinMpsIO19convertBoundToSenseEddRcRdS1_
  Referenced from: /Users/mehdi/Documents/qp/codes/CyClp/CyCoinMpsIO.so
  Expected in: flat namespace
 in /Users/mehdi/Documents/qp/codes/CyClp/CyCoinMpsIO.so

I dont know the reason. I am putting it here for the moment to solve the problem.

*/

/** A quick inlined function to convert from lb/ub style constraint
    definition to sense/rhs/range style */
inline void
ICoinMpsIO::convertBoundToSense(const double lower, const double upper,
					char& sense, double& right,
					double& range) const
{
  range = 0.0;
  if (lower > -infinity_) {
    if (upper < infinity_) {
      right = upper;
      if (upper==lower) {
        sense = 'E';
      } else {
        sense = 'R';
        range = upper - lower;
      }
    } else {
      sense = 'G';
      right = lower;
    }
  } else {
    if (upper < infinity_) {
      sense = 'L';
      right = upper;
    } else {
      sense = 'N';
      right = 0.0;
    }
  }
}


//-----------------------------------------------------------------------------
/** A quick inlined function to convert from sense/rhs/range stryle constraint
    definition to lb/ub style */
inline void
CoinMpsIO::convertSenseToBound(const char sense, const double right,
					const double range,
					double& lower, double& upper) const
{
  switch (sense) {
  case 'E':
    lower = upper = right;
    break;
  case 'L':
    lower = -infinity_;
    upper = right;
    break;
  case 'G':
    lower = right;
    upper = infinity_;
    break;
  case 'R':
    lower = right - range;
    upper = right;
    break;
  case 'N':
    lower = -infinity_;
    upper = infinity_;
    break;
  }
}

