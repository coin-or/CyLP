#include "IClpPackedMatrix.hpp"

void 
IClpPackedMatrix::transposeTimesSubsetAll( IClpSimplex* model, int number,
        const long long int * which,
        const double * COIN_RESTRICT x, double * COIN_RESTRICT y,
        const double * COIN_RESTRICT rowScale, 
        const double * COIN_RESTRICT columnScale,
        double * COIN_RESTRICT spare) const
{
    // get matrix data pointers
    const int *  row = matrix_->getIndices();
    const CoinBigIndex *  columnStart = matrix_->getVectorStarts();
    const double *  elementByColumn = matrix_->getElements();
    if (!spare||!rowScale) {
        if (rowScale) {
            for (int jColumn=0;jColumn<number;jColumn++) {
                int iColumn = which[jColumn];
                CoinBigIndex j;
                CoinBigIndex start=columnStart[iColumn];
                CoinBigIndex next=columnStart[iColumn+1];
                double value=0.0;
                if (iColumn > model->getNumCols()){
                    int jRow = iColumn - model->getNumCols();
                    value = x[jRow]* -1 *rowScale[jRow];
                }
                else{
                    for (j=start;j<next;j++) {
                        int jRow=row[j];
                        value += x[jRow]*elementByColumn[j]*rowScale[jRow];
                    }
                }
                y[iColumn] -= value*columnScale[iColumn];
            }
        } else {
            for (int jColumn=0;jColumn<number;jColumn++) {
                int iColumn = which[jColumn];
                CoinBigIndex j;
                CoinBigIndex start=columnStart[iColumn];
                CoinBigIndex next=columnStart[iColumn+1];
                double value=0.0;
                if (iColumn > model->getNumCols()){
                    int jRow = iColumn - model->getNumCols();
                    value = x[jRow]* -1;
                }
                else{
                    for (j=start;j<next;j++) {
                        int jRow=row[j];
                        value += x[jRow]*elementByColumn[j];
                    }
                }
                y[iColumn] -= value;
            }
        }
  } else {
    // can use spare region
    int iRow;
    int numberRows = matrix_->getNumRows();
    for (iRow=0;iRow<numberRows;iRow++) {
      double value = x[iRow];
      if (value) 
	spare[iRow] = value*rowScale[iRow];
      else
	spare[iRow]=0.0;
    }
    for (int jColumn=0;jColumn<number;jColumn++) {
      int iColumn = which[jColumn];
      CoinBigIndex j;
      CoinBigIndex start=columnStart[iColumn];
      CoinBigIndex next=columnStart[iColumn+1];
      double value=0.0;
      for (j=start;j<next;j++) {
	int jRow=row[j];
	value += spare[jRow]*elementByColumn[j];
      }
      y[iColumn] -= value*columnScale[iColumn];
    }
  }
}

