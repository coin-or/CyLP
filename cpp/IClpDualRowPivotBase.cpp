#include "IClpDualRowPivotBase.h"

int
CppClpDualRowPivotBase::pivotRow()
{

//      assert(model_);
//      int iRow;
//      const int * pivotVariable = model_->pivotVariable();
//      double tolerance = model_->currentPrimalTolerance();
//      // we can't really trust infeasibilities if there is primal error
//      if (model_->largestPrimalError() > 1.0e-8)
//           tolerance *= model_->largestPrimalError() / 1.0e-8;
//      double largest = 0.0;
//      int chosenRow = -1;
//      int numberRows = model_->numberRows();
// #ifdef CLP_DUAL_COLUMN_MULTIPLIER
//      int numberColumns = model_->numberColumns();
// #endif
//      for (iRow = 0; iRow < numberRows; iRow++) {
//           int iSequence = pivotVariable[iRow];
//           double value = model_->solution(iSequence);
//           double lower = model_->lower(iSequence);
//           double upper = model_->upper(iSequence);
// 	  double infeas = CoinMax(value - upper , lower - value);
//      std::cout << "iSequence: " << iSequence ;
//      std::cout << ", lower: " << lower;
//      std::cout << ", upper: " << upper;
//      std::cout << ", value: " << value;
//      std::cout << ", inf: " << infeas << std::endl;
//           if (infeas > tolerance) {
// #ifdef CLP_DUAL_COLUMN_MULTIPLIER
// 	      if (iSequence < numberColumns)
// 		infeas *= CLP_DUAL_COLUMN_MULTIPLIER;
// #endif
// 	      if (infeas > largest) {
// 		if (!model_->flagged(iSequence)) {
// 		  chosenRow = iRow;
// 		  largest = infeas;
// 		}
// 	      }
//           }
//      }
//      std::cout << "chosenRow: " << chosenRow << std::endl << std::endl;
//      return chosenRow;



	//std::cout << "::Cy..Base::pivotRow()...\n";
	//if (this->obj && this->runPivotRow) {
		return this->runPivotRow(this->obj);
	//}
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

     // Do FT update
     model_->factorization()->updateColumnFT(spare, updatedColumn);
     // pivot element
     double alpha = 0.0;
     // look at updated column
     double * work = updatedColumn->denseVector();
     int number = updatedColumn->getNumElements();
     int * which = updatedColumn->getIndices();
     int i;
     int pivotRow = model_->pivotRow();

     if (updatedColumn->packedMode()) {
          for (i = 0; i < number; i++) {
               int iRow = which[i];
               if (iRow == pivotRow) {
                    alpha = work[i];
                    break;
               }
          }
     } else {
          alpha = work[pivotRow];
     }
     return alpha;


	if (this->obj && this->runUpdateWeights) {
		return this->runUpdateWeights(this->obj, input, spare, spare2, updatedColumn);
	}
	std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	<< this->runUpdateWeights << "]\n";
	return -1;
}

void CppClpDualRowPivotBase::updatePrimalSolution(CoinIndexedVector * primalUpdate,
                                       double primalRatio,
                                       double & objectiveChange){

     double * work = primalUpdate->denseVector();
     int number = primalUpdate->getNumElements();
     int * which = primalUpdate->getIndices();
     int i;
     double changeObj = 0.0;
     const int * pivotVariable = model_->pivotVariable();
     if (primalUpdate->packedMode()) {
          for (i = 0; i < number; i++) {
               int iRow = which[i];
               int iPivot = pivotVariable[iRow];
               double & value = model_->solutionAddress(iPivot);
               double cost = model_->cost(iPivot);
               double change = primalRatio * work[i];
               value -= change;
               changeObj -= change * cost;
               work[i] = 0.0;
          }
     } else {
          for (i = 0; i < number; i++) {
               int iRow = which[i];
               int iPivot = pivotVariable[iRow];
               double & value = model_->solutionAddress(iPivot);
               double cost = model_->cost(iPivot);
               double change = primalRatio * work[iRow];
               value -= change;
               changeObj -= change * cost;
               work[iRow] = 0.0;
          }
     }
     primalUpdate->setNumElements(0);
     objectiveChange += changeObj;
     return;

	// if (this->obj && this->runUpdatePrimalSolution) {
	// 	return this->runUpdatePrimalSolution(this->obj, input, theta, &changeInObjective);
	// }
	// std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	// << this->runUpdatePrimalSolution << "]\n";
	// return;

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



