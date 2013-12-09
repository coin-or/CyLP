// Copyright (C) 2004, International Business Machines
// Corporation and others.  All Rights Reserved.
#if defined(_MSC_VER)
// Turn off compiler warning about long names
#  pragma warning(disable:4786)
#endif
#include <cassert>
#include <cmath>
#include <cfloat>
//#define CBC_DEBUG

#include "CbcMessage.hpp"
#include "CbcModel.hpp"
#include "CbcTree.hpp"
#include "CbcCompareUser.hpp"
#include "CoinError.hpp"
#include "CoinHelperFunctions.hpp"

/** Default Constructor

*/
CbcCompareUser::CbcCompareUser(PyObject* obj, runTest_t runTest,
                    runNewSolution_t runNewSolution,
                    runEvery1000Nodes_t runEvery1000Nodes)
  : CbcCompareBase(),
    weight_(-1.0),
    saveWeight_(0.0),
    numberSolutions_(0),
    count_(0),
    treeSize_(0),
    obj(obj),
    runTest(runTest),
    runNewSolution(runNewSolution),
    runEvery1000Nodes(runEvery1000Nodes)
{
  test_=this;
}

// Constructor with weight
//CbcCompareUser::CbcCompareUser (double weight) 
//  : CbcCompareBase(),
//    weight_(weight) ,
//    saveWeight_(0.0),
//    numberSolutions_(0),
//    count_(0),
//    treeSize_(0)
//{
//  test_=this;
//}


// Copy constructor 
CbcCompareUser::CbcCompareUser ( const CbcCompareUser & rhs)
  :CbcCompareBase(rhs)

{
  weight_=rhs.weight_;
  saveWeight_ = rhs.saveWeight_;
  numberSolutions_=rhs.numberSolutions_;
  count_ = rhs.count_;
  treeSize_ = rhs.treeSize_;
  runTest = rhs.runTest;
  runNewSolution = rhs.runNewSolution;
  runEvery1000Nodes = rhs.runEvery1000Nodes;
  obj = rhs.obj;
}

// Clone
CbcCompareBase *
CbcCompareUser::clone() const
{
  return new CbcCompareUser(*this);
}

// Assignment operator 
CbcCompareUser & 
CbcCompareUser::operator=( const CbcCompareUser& rhs)
{

  if (this!=&rhs) {
    CbcCompareBase::operator=(rhs);
    weight_=rhs.weight_;
    saveWeight_ = rhs.saveWeight_;
    numberSolutions_=rhs.numberSolutions_;
    count_ = rhs.count_;
    treeSize_ = rhs.treeSize_;

    runTest = rhs.runTest;
    runNewSolution = rhs.runNewSolution;
    runEvery1000Nodes = rhs.runEvery1000Nodes;
    obj = rhs.obj;
  }
  return *this;
}

// Destructor 
CbcCompareUser::~CbcCompareUser ()
{
}

// Returns true if y better than x
bool 
CbcCompareUser::test (CbcNode * x, CbcNode * y)
{
    return this->runTest(this->obj, (ICbcNode*) x,(ICbcNode*) y);
}
// This allows method to change behavior as it is called
// after each solution
bool 
CbcCompareUser::newSolution(CbcModel * model,
			       double objectiveAtContinuous,
			       int numberInfeasibilitiesAtContinuous) 
{
    return this->runNewSolution(this->obj, (ICbcModel*)model, 
                             objectiveAtContinuous,
                             numberInfeasibilitiesAtContinuous);
}
// This allows method to change behavior 
bool 
CbcCompareUser::every1000Nodes(CbcModel * model, int numberNodes)
{
    return this->runEvery1000Nodes(this->obj, 
                                        (ICbcModel*)model, 
                                        numberNodes);
}
// Returns true if wants code to do scan with alternate criterion
bool 
CbcCompareUser::fullScan() const
{
  return false;
}
// This is alternate test function
bool 
CbcCompareUser::alternateTest (CbcNode * x, CbcNode * y)
{
  // not used
  abort();
  return false;
}
