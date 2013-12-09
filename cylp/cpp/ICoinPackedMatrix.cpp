#include "ICoinPackedMatrix.hpp"

PyObject* ICoinPackedMatrix::np_getIndices(){

	npy_intp dims = this->getNumElements();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, this->IgetIndices() );
	return Arr;
}

PyObject* ICoinPackedMatrix::np_getElements(){

	npy_intp dims = this->getNumElements();
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_DOUBLE, this->IgetElements() );
	return Arr;
}

PyObject* ICoinPackedMatrix::np_getMajorIndices(){

    npy_intp dims = this->getNumElements();
    _import_array();
    PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, this->getMajorIndices() );
    return Arr;
}


PyObject* ICoinPackedMatrix::np_getVectorStarts(){
	npy_intp dims = this->getMajorDim() + 1;
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, this->IgetVectorStarts() );
	return Arr;
}

ICoinPackedMatrix::ICoinPackedMatrix():CoinPackedMatrix()
{
}

ICoinPackedMatrix::ICoinPackedMatrix(const bool colordered,
     const int * rowIndices,
     const int * colIndices,
     const double * elements,
     CoinBigIndex numels ):CoinPackedMatrix(colordered, rowIndices, colIndices, elements, numels){
}

