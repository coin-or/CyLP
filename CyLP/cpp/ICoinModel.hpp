#ifndef ICoinModel_H
#define ICoinModel_H

//#include "ClpModel.hpp"
#include "CoinModel.hpp"
#include "Python.h"
#include <numpy/arrayobject.h>


class ICoinModel: public CoinModel{
public:
	   void addRow(int numberInRow, const int * columns,
	       const double * elements, double rowLower, 
              double rowUpper)
              {CoinModel::addRow(numberInRow, columns, elements, rowLower, rowUpper, NULL);}

		void addColumn(int numberInColumn, const int * rows,
                  const double * elements, 
                  double columnLower, 
                  double columnUpper, double objectiveValue)
                  {CoinModel::addColumn(numberInColumn, rows,elements, columnLower, columnUpper, NULL, false);}

};


#endif