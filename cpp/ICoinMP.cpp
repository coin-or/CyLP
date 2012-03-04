#include "ICoinMP.hpp"
#include <stdio.h>

PCOIN global_pCoin;

int SOLVCALL IterCallback(int    IterCount, 
			double ObjectValue,
			int    IsFeasible, 
			double InfeasValue)
{
	fprintf(stdout, "ITER: iter=%d, obj=%lg, feas=%d, infeas=%lg\n",
		IterCount, ObjectValue, IsFeasible, InfeasValue);
	return 0;
}

int SOLVCALL MipNodeCallback(int    IterCount, 
				int	  MipNodeCount,
				double BestBound,
				double BestInteger,
				int    IsMipImproved)
{
	fprintf(stdout, "NODE: iter=%d, node=%d, bound=%lg, best=%lg, %s\n",
		IterCount, MipNodeCount, BestBound, BestInteger, IsMipImproved ? "Improved" : "");
	return 0;
}

SOLVAPI HPROB SOLVCALL CoinCreateProblem(const char* ProblemName, IClpSimplex* clpmodel)
{
    PCOIN pCoin;

    pCoin = (PCOIN) malloc(sizeof(COININFO));
    global_pCoin = pCoin;
    pCoin->clp = clpmodel; //new ClpSimplex();
    pCoin->clp_presolve = new ClpSolve();
    pCoin->osi = new OsiClpSolverInterface(pCoin->clp);
    pCoin->cbc = NULL;  /* ERRORFIX 2/22/05: Crashes if not NULL when trying to set message handler */
    pCoin->CbcMain0Already = 0;

    pCoin->msghandler = NULL;
    pCoin->iterhandler = NULL;
    pCoin->nodehandler = NULL;

    pCoin->LoadNamesType = SOLV_LOADNAMES_LIST;

    strcpy(pCoin->ProblemName, ProblemName);

    pCoin->ColCount    = 0;
    pCoin->RowCount    = 0;
    pCoin->NZCount     = 0;
    pCoin->RangeCount  = 0;
    pCoin->ObjectSense = 0;
    pCoin->ObjectConst = 0.0;

    pCoin->lenColNamesBuf   = 0;
    pCoin->lenRowNamesBuf   = 0;
    pCoin->lenObjNameBuf = 0;

    pCoin->ObjectCoeffs = NULL;
    pCoin->RHSValues    = NULL;
    pCoin->RangeValues  = NULL;
    pCoin->RowType      = NULL;
    pCoin->MatrixBegin  = NULL;
    pCoin->MatrixCount  = NULL;
    pCoin->MatrixIndex  = NULL;
    pCoin->MatrixValues = NULL;
    pCoin->LowerBounds  = NULL;
    pCoin->UpperBounds  = NULL;
    pCoin->ColNamesBuf  = NULL;
    pCoin->RowNamesBuf  = NULL;
    pCoin->ColNamesList = NULL;
    pCoin->RowNamesList = NULL;
    pCoin->ObjectName   = NULL;

    pCoin->InitValues   = NULL;

    pCoin->RowLower     = NULL;
    pCoin->RowUpper     = NULL;

    pCoin->ColType      = NULL;

    pCoin->SolveAsMIP   = 0;
    pCoin->IntCount     = 0;
    pCoin->BinCount     = 0;
    pCoin->numInts      = 0;
    pCoin->IsInt        = NULL;

    pCoin->SosCount     = 0;
    pCoin->SosNZCount   = 0;
    pCoin->SosType      = NULL;
    pCoin->SosPrior     = NULL;
    pCoin->SosBegin     = NULL;
    pCoin->SosIndex     = NULL;
    pCoin->SosRef       = NULL;
    
    pCoin->PriorCount   = 0;
    pCoin->PriorIndex   = NULL;
    pCoin->PriorValues  = NULL;
    pCoin->BranchDir    = NULL;

    pCoin->SolutionStatus = 0;
    strcpy(pCoin->SolutionText, "");

    pCoin->MessageLogCallback = NULL;
    pCoin->IterationCallback = NULL;
    pCoin->MipNodeCallback = NULL;

    return (HPROB)pCoin;
}

//void RunTestProblem(char* problemName, double optimalValue, int colCount, int rowCount, 
//	  int nonZeroCount, int rangeCount, int objectSense, double objectConst, double* objectCoeffs, 
//	  double* lowerBounds, double* upperBounds, char* rowType, double* rhsValues, double* rangeValues, 
//	  int* matrixBegin, int* matrixCount, int* matrixIndex, double* matrixValues, char** colNames, 
//	  char** rowNames, char* objectName, double* initValues, char* columnType, int LoadNamesType)
void SolveMIP(char* problemName, char* columnType, IClpSimplex* clpmodel)
{
	HPROB hProb;
	int result;
	char filename[260];
    std::cout << "col0 = " << columnType[0] << std::endl;
    std::cout << "col1 = " << columnType[1] << std::endl;
    std::cout << "col2 = " << columnType[2] << std::endl;
    std::cout << "col3 = " << columnType[3] << std::endl;

	fprintf(stdout, "Solve Problem: %s\n", problemName);
	hProb = CoinCreateProblem(problemName, clpmodel);  
//	if (LoadNamesType > 0) {
//		result = CoinSetLoadNamesType(hProb, LoadNamesType);
//	}
//	result = CoinLoadProblem(hProb, colCount, rowCount, nonZeroCount, rangeCount,
//					objectSense, objectConst, objectCoeffs, lowerBounds, upperBounds, 
//					rowType, rhsValues, rangeValues, matrixBegin, matrixCount, 
//					matrixIndex, matrixValues, colNames, rowNames, objectName);
	if (columnType) {
		result = CoinLoadInteger(hProb, columnType);
	}
    result = CoinCheckProblem(hProb);
    if (result != SOLV_CALL_SUCCESS) {
		fprintf(stdout, "Check Problem failed (result = %d)\n", result);
	}
	//result = CoinSetMsgLogCallback(hProb, &MsgLogCallback);
	if (!columnType)
		result = CoinSetIterCallback(hProb, &IterCallback);
	else {
		result = CoinSetMipNodeCallback(hProb, &MipNodeCallback);
	}
	result = CoinOptimizeProblem(hProb, 0);
	//strcpy(filename, problemName);
	//strcat(filename, ".mps");
	//result = CoinWriteFile(hProb, SOLV_FILE_MPS, filename);
	//GetAndCheckSolution(optimalValue, hProb);
	CoinUnloadProblem(hProb);
}

