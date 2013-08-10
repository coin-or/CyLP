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
    // std::cout << "===================================================\n";
    // int n = this->getNumElements();
    // std::cout << "n: " << n << std::endl;
    // std::cout << "MajorDim: " << this->getMajorDim() << std::endl;
    // std::cout << "MinorDim: " << this->getMinorDim() << std::endl;

    // std::cout << "@elements: \n";
    // for (int i = 0; i <= n ; i++)
    // {
    //     std::cout << this->IgetElements()[i] << ", ";
    // }
    // std::cout << std::endl;

    // int* mi = this->getMajorIndices();
    // std::cout << "@major indices: \n";
    // for (int i = 0; i <= n ; i++)
    // {
    //     std::cout << mi[i] << ", ";
    // }
    // delete[] mi;

    // std::cout << std::endl;
    // std::cout << "@indices: \n";
    // for (int i = 0; i <= n ; i++)
    // {
    //     std::cout << this->IgetIndices()[i] << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "@start vecs: \n";
    // for (int i = 0; i <= this->getMajorDim() + 1 ; i++)
    // {
    //     std::cout << this->IgetVectorStarts()[i] << ", ";
    // }
    // std::cout << std::endl;

	npy_intp dims = this->getMajorDim() + 1;
	_import_array();
	PyObject *Arr = PyArray_SimpleNewFromData( 1, &dims, PyArray_INT32, this->IgetVectorStarts() );
	return Arr;
}

ICoinPackedMatrix::ICoinPackedMatrix(){
        CoinPackedMatrix::CoinPackedMatrix();
}

ICoinPackedMatrix::ICoinPackedMatrix(const bool colordered,
     const int * rowIndices,
     const int * colIndices,
     const double * elements,
     CoinBigIndex numels ):CoinPackedMatrix(colordered, rowIndices, colIndices, elements, numels){
}

