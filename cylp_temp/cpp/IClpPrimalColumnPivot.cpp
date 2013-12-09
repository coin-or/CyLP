#include "IClpPrimalColumnPivot.h"

int CppClpPrimalColumnPivotBase::pivotColumn(CoinIndexedVector * updates,
			  CoinIndexedVector * spareRow1,
			  CoinIndexedVector * spareRow2,
			  CoinIndexedVector * spareColumn1,
			  CoinIndexedVector * spareColumn2)
{
std::cout << "PivotColumn should be implemented in a derived class\n";
return -100;
}


CppClpPrimalColumnPivotBase::CppClpPrimalColumnPivotBase(PyObject *obj, RunFct fct) :
  obj(obj),
  fct(fct)
{
}

CppClpPrimalColumnPivotBase::~CppClpPrimalColumnPivotBase()
{
}

void CppClpPrimalColumnPivotBase::saveWeights(ClpSimplex * model,int mode)
{
std::cout << "saveWeight should be implemented in a derived class\n";
}

CppClpPrimalColumnPivotBase* CppClpPrimalColumnPivotBase::clone(bool copyData) const
{
std::cout << "clone should be implemented in a derived class\n";
return 0;
}

