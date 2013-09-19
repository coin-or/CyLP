#include "Python.h"
#include <iostream>
using namespace std;

#include "CglCutGenerator.hpp"
//#include "CoinIndexedVector.hpp"
//#include "IClpSimplex.hpp"
//#include "ClpSimplex.hpp"
//#include "ClpFactorization.hpp"
#include "OsiSolverInterface.hpp"

typedef CglCutGenerator* (*runCglClone_t)(void *instance);

typedef void (*runGenerateCuts_t)(void *instance,
                const OsiSolverInterface *si, OsiCuts *cs, const CglTreeInfo info);


class CppCglCutGeneratorBase : public CglCutGenerator
{
public:
  	PyObject *obj;
	runCglClone_t runCglClone;
	runGenerateCuts_t runGenerateCuts;


	//IClpSimplex model_;

  	CppCglCutGeneratorBase(PyObject *obj, runGenerateCuts_t ,
							 runCglClone_t );
  	virtual ~CppCglCutGeneratorBase();
    CppCglCutGeneratorBase(const CglCutGenerator & source);
    CppCglCutGeneratorBase();

	virtual CglCutGenerator * clone() const;

  virtual void generateCuts(const OsiSolverInterface & si, OsiCuts & cs,
                 const CglTreeInfo info = CglTreeInfo());


};


