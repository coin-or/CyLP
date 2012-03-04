#ifndef ICoinIndexedVector_H
#define ICoinIndexedVector_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "CoinIndexedVector.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>


class ICoinIndexedVector : public CoinIndexedVector{
public:
    ICoinIndexedVector();
	PyObject* getIndicesNPArray();
    PyObject* getDenseVectorNPArray();
	double getItem(int n);
	void setItem(int n, double value);
    void assign(PyObject* ind, PyObject* other);
    void Print(){print();}
};


#endif
