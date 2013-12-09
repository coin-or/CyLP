#include "ICglCutGeneratorBase.h"

void
CppCglCutGeneratorBase::generateCuts(const OsiSolverInterface & si, OsiCuts & cs,
                 const CglTreeInfo info)
{
	//std::cout << "::Cy..Base::generateCuts()...\n";
    if (this->obj && this->runGenerateCuts) {
    	this->runGenerateCuts(this->obj, &si, &cs, info);
        return;
	}
	std::cout << "** generateCuts: invalid cy-state: obj [" << this->obj << "] fct: ["
	          << this->runGenerateCuts << "]\n";
}

CglCutGenerator * CppCglCutGeneratorBase::clone() const {
	//std::cout << "::Cy..Base::clone()...\n";
	if (this->obj && this->runCglClone) {
		return this->runCglClone(this->obj);
	}
	std::cerr << "** clone: invalid cy-state: obj [" << this->obj << "] fct: ["
	          << this->runCglClone << "]\n";
	return NULL;
}


CppCglCutGeneratorBase::CppCglCutGeneratorBase(PyObject *obj, runGenerateCuts_t runGenerateCuts,
													   runCglClone_t runCglClone) :
  obj(obj),
	runCglClone(runCglClone),
  runGenerateCuts(runGenerateCuts)
{
}

CppCglCutGeneratorBase::~CppCglCutGeneratorBase()
{
}


CppCglCutGeneratorBase::CppCglCutGeneratorBase(const CglCutGenerator & source):
    CglCutGenerator(source),
    obj(obj),
    runCglClone(runCglClone),
    runGenerateCuts(runGenerateCuts)
{
}


CppCglCutGeneratorBase::CppCglCutGeneratorBase():
    CglCutGenerator(),
    obj(obj),
    runCglClone(runCglClone),
    runGenerateCuts(runGenerateCuts)
{
}

