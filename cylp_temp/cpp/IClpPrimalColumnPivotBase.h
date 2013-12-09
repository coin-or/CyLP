#include "Python.h"
#include <iostream>
using namespace std;

#include "ClpPrimalColumnPivot.hpp"
#include "CoinIndexedVector.hpp"
#include "IClpSimplex.hpp"
//#include "ClpSimplex.hpp"
#include "ClpFactorization.hpp"

typedef int (*runPivotColumn_t)(void *instance, CoinIndexedVector*,CoinIndexedVector*
					  ,CoinIndexedVector*,CoinIndexedVector*,CoinIndexedVector*);

typedef ClpPrimalColumnPivot* (*runClone_t)(void *instance, bool copyData);

typedef void (*runSaveWeights_t)(void *instance, IClpSimplex * model,int mode);



class CppClpPrimalColumnPivotBase : public ClpPrimalColumnPivot
{
public:
  	PyObject *obj;
  	runPivotColumn_t runPivotColumn;
	runClone_t runClone;
	runSaveWeights_t runSaveWeights;

	//IClpSimplex model_;

  	CppClpPrimalColumnPivotBase(PyObject *obj, runPivotColumn_t ,
							 runClone_t , runSaveWeights_t );
  	virtual ~CppClpPrimalColumnPivotBase();

	virtual ClpPrimalColumnPivot * clone(bool copyData = true) const;
	//virtual void saveWeights(IClpSimplex * model,int mode);
  	virtual void saveWeights(ClpSimplex * model,int mode);
  	virtual int pivotColumn(CoinIndexedVector * updates,
			  CoinIndexedVector * spareRow1,
			  CoinIndexedVector * spareRow2,
			  CoinIndexedVector * spareColumn1,
						  CoinIndexedVector * spareColumn2);
	void setModel(IClpSimplex* m);
	IClpSimplex* model();

};


