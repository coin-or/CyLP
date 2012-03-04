#ifndef IClpSimplex_H
#define IClpSimplex_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "ClpSimplex.hpp"
#include "ClpPresolve.hpp"
#include "ClpLinearObjective.hpp"
#include "CoinIndexedVector.hpp"
#include "ClpFactorization.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "ICbcModel.hpp"

//#include "ClpSimplexPrimal.hpp"

typedef int (*runIsPivotAcceptable_t)(void *instance);
typedef int (*varSelCriteria_t)(void *instance, int varInd);


class IClpSimplex : public ClpSimplex{
public:
    IClpSimplex(const ClpSimplex &rhs,PyObject *obj,
                            runIsPivotAcceptable_t runPivotRow,
                            varSelCriteria_t RunVarSelCriteria,
                            int useCustomPrimal, int scalingMode=-1 );
	IClpSimplex(PyObject *obj, runIsPivotAcceptable_t runPivotRow,
                varSelCriteria_t RunVarSelCriteria );
	// For initializing instances that are build using ClpSimplex consructor (NOT IClpSimplex)
    void setCriteria(varSelCriteria_t vsc);

    int argWeightedMax(PyObject* arr, PyObject* whr, double weight);
    int argWeightedMax(PyObject* arr, PyObject* arr_ind, PyObject* w, PyObject* w_ind);

    PyObject *obj;
  	runIsPivotAcceptable_t runIsPivotAcceptable;
	int isPivotAcceptable();
    
    varSelCriteria_t varSelCriteria;
    int checkVar(int varInd);	
    PyObject* filterVars(PyObject* inds);
    int* tempIntArray;
    bool tempArrayExists;
    void createTempArray(void);
    
    int* QP_ComplementarityList;
	int* QP_BanList;
	int QP_ExistsBannedVariable;
	
	void setComplementarityList(int *);
	int* ComplementarityList();

    bool varIsFree(int ind);
    bool varBasic(int ind);
    bool varAtUpperBound(int ind);
    bool varAtLowerBound(int ind);
    bool varSuperBasic(int ind);
    bool varIsFixed(int ind);
	PyObject * getReducedCosts();
    PyObject * getStatusArray();
	PyObject * getComplementarityList();
	PyObject * getPivotVariable();
	PyObject * getPrimalRowSolution();
	PyObject * getPrimalColumnSolution();
	PyObject * getDualRowSolution();
	PyObject * getDualColumnSolution();
	
	void getBInvACol(int col, double* vec);
	void getACol(int ncol, CoinIndexedVector * colArray);
	
	int customPrimal;
	void useCustomPrimal(int);
	int getUseCustomPrimal();
	
	
	void setPrimalColumnPivotAlgorithm(ClpPrimalColumnPivot *choice){ClpSimplex::setPrimalColumnPivotAlgorithm(*choice);}
	
	
	int loadProblem (CoinModel * modelObject,bool tryPlusMinusOne=false){return ClpSimplex::loadProblem(*modelObject, tryPlusMinusOne);}
	//double* infeasibilityRay();
	
	int primal(int ifValuesPass=0, int startFinishOptions=0);
	
	IClpSimplex (ClpSimplex * wholeModel,
	      int numberColumns, const int * whichColumns);
	      
	~IClpSimplex();
	
	void dualExpanded(ClpSimplex * model,CoinIndexedVector * array, double * other,int mode);
	
	
	void convertBoundToSense(const double lower, const double upper,
						 double& right);
	void extractSenseRhsRange(double* rhs_);
	void getRightHandSide(double* righthandside); 
	
	
	double* tempRow;
	CoinIndexedVector* tempRow_vector;
	
	void vectorTimesB_1(CoinIndexedVector*);
    void transposeTimesSubsetAll(int number, long long int* which, double* pi, double* y);
    void transposeTimesSubset(int number, int* which, double* pi, double* y);
    	
	IClpSimplex*	preSolve(IClpSimplex* si,
                              double feasibilityTolerance=0.0,
                              bool keepIntegers=true,
                              int numberPasses=5,
                              bool dropNames=false,
                              bool doRowObjective=false);
                              
    void setComplement(int var1, int var2){QP_ComplementarityList[var1] = var2; QP_ComplementarityList[var2] = var1;}
    
    
    inline double getCoinInfinity(){return COIN_DBL_MAX;}
    
    inline void setColumnUpperArray(double *cu){columnUpper_ = cu;}
    inline void setColumnLowerArray(double *cl){columnLower_ = cl;}
    inline void setRowUpperArray(double *ru){rowUpper_ = ru;}
    inline void setRowLowerArray(double *rl){rowLower_ = rl;}
    
    inline void setObjectiveArray(double *o, int numberColumns)
    {
    if (objective_)
    	delete objective_;
    objective_ = new ClpLinearObjective(o, numberColumns);
    }
    
	void setVariableName(int varInd,  char* name);
    		
    /// Partial pricing 
    int partialPrice(int start, int end, int* numberWanted)
      {
       int bestVarInd;
       this->clpMatrix()->partialPricing(this,
                                          static_cast<double>(start), 
                                          static_cast<double>(end), 
                                          bestVarInd, 
                                          *numberWanted);
        return bestVarInd;
        }

    ICbcModel* getICbcModel();

};

double cdot(CoinIndexedVector* v1, CoinIndexedVector* v2);


#endif
