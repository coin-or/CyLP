#include "IClpDualRowPivotBase.h"
#include "ICoinIndexedVector.hpp"

int
CppClpDualRowPivotBase::pivotRow()
{
	//std::cout << "::Cy..Base::pivotRow()...\n";
	if (this->obj && this->runPivotRow) {
		return this->runPivotRow(this->obj);
	}
	std::cerr << "** pivotRow: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runPivotRow << "]\n";
	return -100;
}

ClpDualRowPivot * CppClpDualRowPivotBase::clone(bool copyData) const {
	//std::cout << "::Cy..Base::clone()...\n";
	if (this->obj && this->runDualPivotClone) {
		return this->runDualPivotClone(this->obj,copyData);
	}
	std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runDualPivotClone << "]\n";
	return NULL;
}

double CppClpDualRowPivotBase::updateWeights(CoinIndexedVector * input,
                                  CoinIndexedVector * spare,
                                  CoinIndexedVector * spare2,
                                  CoinIndexedVector * updatedColumn) {

//     // Do FT update
//     model_->factorization()->updateColumnFT(spare, updatedColumn);
//     static_cast<ICoinIndexedVector*>(updatedColumn)->Print();
//     // pivot element
//     double alpha = 0.0;
//     // look at updated column
//     double * work = updatedColumn->denseVector();
//     int number = updatedColumn->getNumElements();
//     int * which = updatedColumn->getIndices();
//     int i;
//     int pivotRow = model_->pivotRow();
//     if (updatedColumn->packedMode()) {
//          for (i = 0; i < number; i++) {
//               int iRow = which[i];
//               if (iRow == pivotRow) {
//                    alpha = work[i];
//                    break;
//               }
//          }
//     } else {
//          alpha = work[pivotRow];
//     }
//     std::cout << "pr: " << pivotRow << ", alpha: " << alpha << std::endl;
//     return alpha;
//

	if (this->obj && this->runUpdateWeights) {
		return this->runUpdateWeights(this->obj, input, spare, spare2, updatedColumn);
	}
	std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runUpdateWeights << "]\n";
	return -1;
}

void CppClpDualRowPivotBase::updatePrimalSolution(
									   CoinIndexedVector * primalUpdate,
                                       double primalRatio,
                                       double & objectiveChange){

//     double * work = primalUpdate->denseVector();
//     int number = primalUpdate->getNumElements();
//     int * which = primalUpdate->getIndices();
//     int i;
//     double changeObj = 0.0;
//     const int * pivotVariable = model_->pivotVariable();
//     std::cout << "Before: \n" ;
//     static_cast<ICoinIndexedVector*>(primalUpdate)->Print();
//     if (primalUpdate->packedMode()) {
//          for (i = 0; i < number; i++) {
//               int iRow = which[i];
//               int iPivot = pivotVariable[iRow];
//               double & value = model_->solutionAddress(iPivot);
//               double cost = model_->cost(iPivot);
//               double change = primalRatio * work[i];
//               value -= change;
//               changeObj -= change * cost;
//               std::cout << "p  : change: " << change << ", cost: " << cost << std::endl;
//               work[i] = 0.0;
//          }
//     } else {
//          for (i = 0; i < number; i++) {
//               int iRow = which[i];
//               int iPivot = pivotVariable[iRow];
//               double & value = model_->solutionAddress(iPivot);
//               double cost = model_->cost(iPivot);
//               double change = primalRatio * work[iRow];
//               value -= change;
//               changeObj -= change * cost;
//               std::cout << "unp: change: " << change << ", cost: " << cost << std::endl;
//               work[iRow] = 0.0;
//          }
//     }
//     std::cout << "After: \n" ;
//     static_cast<ICoinIndexedVector*>(primalUpdate)->Print();
//     primalUpdate->setNumElements(0);
//     std::cout << "change: " << changeObj << std::endl;
//
//     objectiveChange += changeObj;
//
//     std::cout << "objective change: " << objectiveChange << std::endl;
//
//
//     return;
	 if (this->obj && this->runUpdatePrimalSolution) {
    	return this->runUpdatePrimalSolution(this->obj, primalUpdate,
     											primalRatio, &objectiveChange);
     	 }
     std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
     << this->runUpdatePrimalSolution << "]\n";
     return;

}


CppClpDualRowPivotBase::CppClpDualRowPivotBase(PyObject *obj, runPivotRow_t runPivotRow,
													   runDualPivotClone_t runDualPivotClone, runUpdateWeights_t runUpdateWeights,
													   runUpdatePrimalSolution_t runUpdatePrimalSolution) :
  obj(obj),
  runPivotRow(runPivotRow),
	runDualPivotClone(runDualPivotClone),
	runUpdateWeights(runUpdateWeights),
	runUpdatePrimalSolution(runUpdatePrimalSolution)
{
}

CppClpDualRowPivotBase::~CppClpDualRowPivotBase()
{
}

void CppClpDualRowPivotBase::setModel(IClpSimplex* m)
{
	ClpSimplex* s = static_cast<ClpSimplex*>(m);
	model_ = s;
}

IClpSimplex* CppClpDualRowPivotBase::model()
{
	return static_cast<IClpSimplex*> (model_);
}



