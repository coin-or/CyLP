#ifndef IClpPackedMatrix_H
#define IClpPackedMatrix_H

//#define NPY_NO_DEPRECATED_API

//#include "ClpModel.hpp"
#include "ClpPackedMatrix.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>
#include "CoinFinite.hpp"
#include "CoinPragma.hpp"
#include "IClpSimplex.hpp"

class IClpPackedMatrix : public ClpPackedMatrix{
public:
    IClpPackedMatrix();
   
    void transposeTimesSubsetAll(IClpSimplex* model,  int number,
        const long long int * which,
        const double * COIN_RESTRICT x, double *  y,
        const double *  rowScale, 
        const double *  columnScale,
        double *  spare) const;

};


#endif
