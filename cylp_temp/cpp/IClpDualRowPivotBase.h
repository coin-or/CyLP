#include "Python.h"
#include <iostream>
using namespace std;

#include "ClpDualRowPivot.hpp"
#include "CoinIndexedVector.hpp"
#include "IClpSimplex.hpp"
//#include "ClpSimplex.hpp"
#include "ClpFactorization.hpp"

typedef int (*runPivotRow_t)(void *instance);

typedef ClpDualRowPivot* (*runDualPivotClone_t)(void *instance, bool copyData);

typedef double (*runUpdateWeights_t)(void *instance,
                                  CoinIndexedVector * input,
                                  CoinIndexedVector * spare,
                                  CoinIndexedVector * spare2,
                                  CoinIndexedVector * updatedColumn);

typedef void (*runUpdatePrimalSolution_t)(void *instance,
                                       CoinIndexedVector * input,
                                       double theta,
                                       double * changeInObjective);



class CppClpDualRowPivotBase : public ClpDualRowPivot
{
public:
  	PyObject *obj;
  	runPivotRow_t runPivotRow;
	runDualPivotClone_t runDualPivotClone;
	runUpdateWeights_t runUpdateWeights;
    runUpdatePrimalSolution_t runUpdatePrimalSolution;

	//IClpSimplex model_;

  	CppClpDualRowPivotBase(PyObject *obj, runPivotRow_t ,
							 runDualPivotClone_t , runUpdateWeights_t, runUpdatePrimalSolution_t );
  	virtual ~CppClpDualRowPivotBase();

	virtual ClpDualRowPivot * clone(bool copyData = true) const;
	//virtual void saveWeights(IClpSimplex * model,int mode);
  	virtual double updateWeights(CoinIndexedVector * input,
                                  CoinIndexedVector * spare,
                                  CoinIndexedVector * spare2,
                                  CoinIndexedVector * updatedColumn);

  	virtual void updatePrimalSolution(CoinIndexedVector * input,
                                       double theta,
                                       double& changeInObjective);

    virtual int pivotRow();

	void setModel(IClpSimplex* m);
	IClpSimplex* model();
};


