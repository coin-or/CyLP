#include "ICglCutGeneratorBase.h"

void
CppCglCutGeneratorBase::generateCuts(const OsiSolverInterface & si, OsiCuts & cs,
                 const CglTreeInfo info) const
{
	std::cout << "::Cy..Base::generateCuts()...\n";
    std::cout << "This: " << this << "\n";
	if (this->obj && this->runGenerateCuts) {
        std::cout << "Everything seems good:\n";
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
  runGenerateCuts(runGenerateCuts),
	runCglClone(runCglClone)
{
        std::cout << "ccccccccccccccccccccccccccccccccc0\n\n";

}

CppCglCutGeneratorBase::~CppCglCutGeneratorBase()
{
}


CppCglCutGeneratorBase::CppCglCutGeneratorBase(const CglCutGenerator & source):
    CglCutGenerator(source),
    obj(obj),
    runGenerateCuts(runGenerateCuts),
    runCglClone(runCglClone)
{
    std::cout << "ccccccccccccccccccccccccccccccccc1\n\n";
}


CppCglCutGeneratorBase::CppCglCutGeneratorBase():
    CglCutGenerator(),
    obj(obj),
    runGenerateCuts(runGenerateCuts),
    runCglClone(runCglClone)
{
        std::cout << "ccccccccccccccccccccccccccccccccc2\n\n";

}

