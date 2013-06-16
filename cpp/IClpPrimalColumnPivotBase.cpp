#include "IClpPrimalColumnPivotBase.h"
#include "ICoinIndexedVector.hpp"

int
CppClpPrimalColumnPivotBase::pivotColumn(CoinIndexedVector* updates, CoinIndexedVector* spareRow1,
								   CoinIndexedVector* spareRow2, CoinIndexedVector* spareColumn1,
								   CoinIndexedVector* spareColumn2 )
{
	//std::cout << "::Cy..Base::pivotColumn()...\n";
	if (this->obj && this->runPivotColumn) {
		return this->runPivotColumn(this->obj, updates, spareRow1, spareRow2, spareColumn1, spareColumn2);
	}
	std::cerr << "** pivotColumn: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runPivotColumn << "]\n";
	return -100;
}

ClpPrimalColumnPivot * CppClpPrimalColumnPivotBase::clone(bool copyData) const {
	//std::cout << "::Cy..Base::clone()...\n";
	if (this->obj && this->runClone) {
		return this->runClone(this->obj,copyData);
	}
	std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runClone << "]\n";
	return NULL;
}

void CppClpPrimalColumnPivotBase::saveWeights(ClpSimplex * model,int mode)
{
	IClpSimplex* m = static_cast<IClpSimplex*>(model);
	if (this->obj && this->runSaveWeights) {
		this->runSaveWeights(this->obj,m, mode);
	return;
	}
	std::cerr << "** saveWeights: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runSaveWeights << "]\n";
	return;
}

CppClpPrimalColumnPivotBase::CppClpPrimalColumnPivotBase(PyObject *obj, runPivotColumn_t runPivotColumn,
													   runClone_t runClone, runSaveWeights_t runSaveWeights) :
  obj(obj),
  runPivotColumn(runPivotColumn),
	runClone(runClone),
	runSaveWeights(runSaveWeights)
{
}

CppClpPrimalColumnPivotBase::~CppClpPrimalColumnPivotBase()
{
}

void CppClpPrimalColumnPivotBase::setModel(IClpSimplex* m)
{
	ClpSimplex* s = static_cast<ClpSimplex*>(m);
	model_ = s;
}

IClpSimplex* CppClpPrimalColumnPivotBase::model()
{
	return static_cast<IClpSimplex*> (model_);
}

