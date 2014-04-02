#ifndef IClpSimplex_H
#define IClpSimplex_H

//#define NPY_NO_DEPRECATED_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#include "ClpModel.hpp"
#include "ClpSimplex.hpp"
#include "ClpPresolve.hpp"
#include "ClpLinearObjective.hpp"
#include "CoinIndexedVector.hpp"
#include "ClpFactorization.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "ICbcModel.hpp"
#include "ClpParameters.hpp"

#include "ICoinPackedMatrix.hpp"


//#include "ClpSimplexPrimal.hpp"

typedef int (*runIsPivotAcceptable_t)(void *instance);
typedef int (*varSelCriteria_t)(void *instance, int varInd);


class IClpSimplex : public ClpSimplex{
public:
    ClpPresolve pinfo;
    IClpSimplex(const ClpSimplex &rhs,PyObject *obj,
                            runIsPivotAcceptable_t runPivotRow,
                            varSelCriteria_t RunVarSelCriteria,
                            int useCustomPrimal, int scalingMode=-1 );
	IClpSimplex(PyObject *obj, runIsPivotAcceptable_t runPivotRow,
                varSelCriteria_t RunVarSelCriteria );
    int initialSolve(int preSolveType=ClpSolve::presolveOn);
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
	void setReducedCosts(double*);
    PyObject * getStatusArray();
	PyObject * getComplementarityList();
	PyObject * getPivotVariable();
	PyObject * getPrimalRowSolution();
	PyObject * getPrimalColumnSolution();
    PyObject * getPrimalColumnSolutionAll();
    PyObject * getSolutionRegion();
    PyObject * getCostRegion();
	PyObject * getDualRowSolution();
	PyObject * getDualColumnSolution();
	PyObject * getObjective();
	PyObject * getRowLower();
	PyObject * getRowUpper();
	PyObject * getColLower();
    PyObject * getColUpper();
    PyObject * getColumnScale();
    PyObject * getRowScale();


    PyObject* getLower();
    PyObject* getUpper();

    PyObject* getIntegerInformation();




	void getBInvACol(int col, double* vec);
	void getACol(int ncol, CoinIndexedVector * colArray);
  int updateColumnFT(CoinIndexedVector * spare,
                          CoinIndexedVector * updatedColumn)
      {
      return this->factorization()->updateColumnFT(spare, updatedColumn);
      }

  int updateColumnTranspose (CoinIndexedVector * regionSparse,
                                 CoinIndexedVector * regionSparse2){
      return this->factorization()->updateColumnTranspose(regionSparse, regionSparse2);
      }

	int customPrimal;
	void useCustomPrimal(int);
	int getUseCustomPrimal();


	void setPrimalColumnPivotAlgorithm(ClpPrimalColumnPivot *choice){ClpSimplex::setPrimalColumnPivotAlgorithm(*choice);}
  void setDualRowPivotAlgorithm(ClpDualRowPivot *choice){ClpSimplex::setDualRowPivotAlgorithm(*choice);}

    void loadQuadraticObjective(const CoinPackedMatrix* matrix){ClpModel::loadQuadraticObjective(*matrix);}
    ICoinPackedMatrix* getMatrix(){return static_cast<ICoinPackedMatrix*>(ClpModel::matrix());}

	int loadProblem (CoinModel * modelObject,bool tryPlusMinusOne=false){return ClpSimplex::loadProblem(*modelObject, tryPlusMinusOne);}
	//double* infeasibilityRay();
    void loadProblem (const CoinPackedMatrix* matrix,
		     const double* collb, const double* colub,
		     const double* obj,
		     const double* rowlb, const double* rowub,
		     const double * rowObjective=NULL){
                   ClpSimplex::loadProblem(*matrix, collb, colub, obj, rowlb, rowub, rowObjective);}

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
    void transposeTimes(const ClpSimplex * model, double scalar,
                                 const CoinIndexedVector * x,
                                 CoinIndexedVector * y,
                                 CoinIndexedVector * z);

    IClpSimplex*	preSolve(IClpSimplex* si,
                              double feasibilityTolerance=0.0,
                              bool keepIntegers=true,
                              int numberPasses=5,
                              bool dropNames=false,
                              bool doRowObjective=false);
    void postSolve(bool updateStatus=true);

    int dualWithPresolve(IClpSimplex* si,
                              double feasibilityTolerance=0.0,
                              bool keepIntegers=true,
                              int numberPasses=5,
                              bool dropNames=false,
                              bool doRowObjective=false);

    int primalWithPresolve(IClpSimplex* si,
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

    inline void setColumnUpperSubset(int n, int *indicesOfIndices, int *indices, double* values){
        for (int i = 0 ; i < n ; i++)
            setColumnUpper(indices[indicesOfIndices[i]], values[indicesOfIndices[i]]);
    }
    inline void setColumnLowerSubset(int n, int *indicesOfIndices, int *indices, double* values){
        for (int i = 0 ; i < n ; i++)
            setColumnLower(indices[indicesOfIndices[i]], values[indicesOfIndices[i]]);
    }

    inline void setColumnUpperFirstElements(int n, double* values){
        for (int i = 0 ; i < n ; i++)
            setColumnUpper(i, values[i]);
    }
    inline void setColumnLowerFirstElements(int n, double* values){
        for (int i = 0 ; i < n ; i++)
            setColumnLower(i, values[i]);
    }

    inline void setObjectiveArray(double *o, int numberColumns)
    {
    if (objective_)
    	delete objective_;
    objective_ = new ClpLinearObjective(o, numberColumns);
    }

	void setVariableName(int varInd,  char* name);
    void setConstraintName(int constInd,  char* name);
    std::vector<std::string> getVariableNames();

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
    void writeLp(const char *filename,
                       const char *extension = "lp",
                       double epsilon = 1e-5,
                       int numberAcross = 10,
                       int decimals = 5,
                       double objSense = 0.0,
                       bool useRowNames = true);

    void setBasisStatus(const int* cstat, const int* rstat);
    void getBasisStatus(int* cstat, int* rstat);

    void setMaxNumIteration(int m);

};

double cdot(CoinIndexedVector* v1, CoinIndexedVector* v2);


#endif
