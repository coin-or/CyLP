#include "ClpPrimalColumnPivot.hpp"
#include "CoinIndexedVector.hpp"
#include "Python.h"

#include <iostream>
 
typedef void (*RunFct)(void *instance);

class CppClpPrimalColumnPivotBase : public ClpPrimalColumnPivot
{
public:
  PyObject *obj;
  RunFct fct;

  CppClpPrimalColumnPivotBase(PyObject *obj, RunFct fct);
  virtual ~CppClpPrimalColumnPivotBase();
 
virtual int pivotColumn(CoinIndexedVector * updates,
			  CoinIndexedVector * spareRow1,
			  CoinIndexedVector * spareRow2,
			  CoinIndexedVector * spareColumn1,
			  CoinIndexedVector * spareColumn2);

virtual void maximumPivotsChanged() {std::cout << "c++: maximumPivot called\n"; }

virtual void maximumPivotsChanged_impl() {std::cout << "c++: maximumPivot called\n"; }

virtual void saveWeights(ClpSimplex * model,int mode);

virtual CppClpPrimalColumnPivotBase* clone(bool copyData = true) const;

};
