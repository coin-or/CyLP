#include "IClpPrimalColumnPivotBase.h"

int
CppClpPrimalColumnPivotBase::pivotColumn(CoinIndexedVector* v1, CoinIndexedVector* v2, 
								   CoinIndexedVector* v3, CoinIndexedVector* v4, CoinIndexedVector* v5 )
{
	//std::cout << "::Cy..Base::pivotColumn()...\n";
	//if (this->obj && this->runPivotColumn) {
		return this->runPivotColumn(this->obj,v1,v2,v3,v4,v5);
	//}
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



// Returns pivot column, -1 if none
int CppClpPrimalColumnPivotBase::DantzigDualUpdate(CoinIndexedVector * updates,
				CoinIndexedVector * spareRow1,
			  CoinIndexedVector * spareRow2,
			  CoinIndexedVector * spareColumn1,
			  CoinIndexedVector * spareColumn2)
{
	assert(model_);
	int iSection,j;
	int number;
	int * index;
	double * updateBy;
	double * reducedCost;
	
	bool anyUpdates;
	
	if (updates->getNumElements()) {
		anyUpdates=true;
	} else {
		// sub flip - nothing to do
		anyUpdates=false;
	}
	if (anyUpdates) {
        //std::cout << "any updates\n";
		model_->factorization()->updateColumnTranspose(spareRow2,updates);
		// put row of tableau in rowArray and columnArray
		model_->clpMatrix()->transposeTimes(model_,-1.0,
											updates,spareColumn2,spareColumn1);
		for (iSection=0;iSection<2;iSection++) {
			
			reducedCost=model_->djRegion(iSection);
			
			if (!iSection) {
				number = updates->getNumElements();
				index = updates->getIndices();
				updateBy = updates->denseVector();
			} else {
				number = spareColumn1->getNumElements();
				index = spareColumn1->getIndices();
				updateBy = spareColumn1->denseVector();
			}
			
			for (j=0;j<number;j++) {
				int iSequence = index[j];
				double value = reducedCost[iSequence];
				value -= updateBy[j];
				updateBy[j]=0.0;
				reducedCost[iSequence] = value;
			}
			
		}
		updates->setNumElements(0);
		spareColumn1->setNumElements(0);
	}
	
	
	return 0;
 } 
