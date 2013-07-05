#include "IClpSimplex.hpp"

#include "CoinHelperFunctions.hpp"
#include "CoinMessageHandler.hpp"

#include "ClpPrimalColumnSteepest.hpp"
#include "ClpDualRowSteepest.hpp"
#include "ClpEventHandler.hpp"

#include "OsiSolverInterface.hpp"
#include "OsiClpSolverInterface.hpp"

#include "CbcModel.hpp"
#include "CbcSolver.hpp"
#include "CbcEventHandler.hpp"
#include "CbcBranchActual.hpp"

#include "CglProbing.hpp"
#include "CglGomory.hpp"
#include "CglKnapsackCover.hpp"
#include "CglOddHole.hpp"
#include "CglClique.hpp"
#include "CglLiftAndProject.hpp"
#include "CglSimpleRounding.hpp"

#include "CoinMP.h"

class  CBIterHandler;
class CBMessageHandler;
class CBNodeHandler;

typedef struct {
                ClpSimplex *clp;
                ClpSolve *clp_presolve;
                OsiClpSolverInterface *osi;
                CbcModel *cbc;
                int CbcMain0Already;

                CBMessageHandler *msghandler;
                CBIterHandler *iterhandler;
                CBNodeHandler *nodehandler;

                CglProbing *probing;
                CglGomory *gomory;
                CglKnapsackCover *knapsack;
                CglOddHole *oddhole;
                CglClique *clique;
                CglLiftAndProject *liftpro;
                CglSimpleRounding *rounding;

                int LoadNamesType;

                char ProblemName[200];

                int ColCount;
                int RowCount;
                int NZCount;
                int RangeCount;
                int ObjectSense;
                double ObjectConst;

                int lenColNamesBuf;
                int lenRowNamesBuf;
                int lenObjNameBuf;

                double* ObjectCoeffs;
                double* RHSValues;
                double* RangeValues;
                char* RowType;
                int* MatrixBegin;
                int* MatrixCount;
                int* MatrixIndex;
                double* MatrixValues;
                double* LowerBounds;
                double* UpperBounds;
                char* ColNamesBuf;
                char* RowNamesBuf;
                char** ColNamesList;
                char** RowNamesList;
                char* ObjectName;

                double* InitValues;

                double* RowLower;
                double* RowUpper;

                char* ColType;

                int SolveAsMIP;
                int IntCount;
                int BinCount;
                int numInts;
                char* IsInt;

                int SosCount;
                int SosNZCount;
                int* SosType;
                int* SosPrior;
                int* SosBegin;
                int* SosIndex;
                double* SosRef;

                int PriorCount;
                int* PriorIndex;
                int* PriorValues;
                int* BranchDir;

                int SolutionStatus;
                char SolutionText[200];

                MSGLOGCALLBACK  MessageLogCallback;
                ITERCALLBACK    IterationCallback;
                MIPNODECALLBACK MipNodeCallback;
                } COININFO, *PCOIN;


void SolveMIP(char* problemName, char* columnType, IClpSimplex* clpModel);
