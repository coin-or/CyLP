#ifndef ICoinPackedMatrix_H
#define ICoinPackedMatrix_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "CoinPackedMatrix.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "CoinFinite.hpp"

class ICoinPackedMatrix : public CoinPackedMatrix{
public:
	PyObject* np_getIndices();
	PyObject* np_getElements();
	PyObject* np_getVectorStarts();
    PyObject* np_getMajorIndices();

	inline int * IgetIndices() const { return index_; }
	inline double * IgetElements() const { return element_; }
	inline CoinBigIndex * IgetVectorStarts() const { return start_; }

	ICoinPackedMatrix();

	ICoinPackedMatrix(const bool colordered,
     const int * rowIndices,
     const int * colIndices,
     const double * elements,
     CoinBigIndex numels );
};


#endif
