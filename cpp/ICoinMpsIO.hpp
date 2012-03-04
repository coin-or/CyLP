#ifndef ICoinMpsIO_H
#define ICoinMpsIO_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "CoinMpsIO.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "ICoinPackedMatrix.hpp"


class ICoinMpsIO : public CoinMpsIO{
public:

	~ICoinMpsIO();
	ICoinMpsIO();
	
	int* d_colStart;
	int* d_cols;
	double* d_elements;
	
	PyObject* getQPColumnStarts();
	PyObject* getQPColumns();
	PyObject* getQPElements();
	
	PyObject* np_getColLower();
	PyObject* np_getColUpper();
	PyObject* np_getRowSense();
	PyObject* np_getRightHandSide();
	PyObject* np_getRowRange();
	PyObject* np_getRowLower();
	PyObject* np_getRowUpper();
	PyObject* np_getObjCoefficients();
	PyObject* np_integerColumns();
	PyObject* np_rowName(int index);
	PyObject* np_columnName(int index);


	double * IRowLower() const;
	double * IRowUpper() const;
	double * IColLower() const;
	double * IColUpper() const;
	char * IRowSense() const;
	double * IRightHandSide() const;
	double * IRowRange() const;
	char * IintegerColumns() const;
	double * IObjCoefficients() const;
	
	double getObjectiveOffset();
	
	inline void
    convertSenseToBound(const char sense, const double right,
			const double range,
			double& lower, double& upper) const;
			
	inline void
    convertBoundToSense(const double lower, const double upper,
			char& sense, double& right, double& range) const;

	int IreadQuadraticMps(const char * filename, int checkSymmetry);
	
	ICoinPackedMatrix * IgetMatrixByRow() const;
	ICoinPackedMatrix * IgetMatrixByCol() const;
};


#endif
