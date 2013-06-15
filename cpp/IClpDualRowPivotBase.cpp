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
     if (this->obj && this->runUpdatePrimalSolution) {
        return this->runUpdatePrimalSolution(this->obj, primalUpdate,
                                                primalRatio, &objectiveChange);
         }
     std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
     << this->runUpdatePrimalSolution << "]\n";
     return;

}


CppClpDualRowPivotBase::CppClpDualRowPivotBase(PyObject *obj,
                                    runPivotRow_t runPivotRow,
                                    runDualPivotClone_t runDualPivotClone,
                                    runUpdateWeights_t runUpdateWeights,
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



