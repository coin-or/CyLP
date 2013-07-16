cimport numpy as np

cdef extern from "CoinModel.hpp":
    cdef cppclass CppCoinModel "CoinModel":

        void addRow(int numberInRow, int * columns,
                        double * elements, double rowLower,
                        double rowUpper, char * name)

        void addRow(int numberInRow, int * columns,
                        double * elements, double rowLower,
                        double rowUpper)

        void addColumn(int numberInColumn, int * rows,
                double * elements,
                double columnLower,
                double columnUpper, double objectiveValue,
                char * name, int isInteger)

        void addColumn(int numberInColumn, int * rows,
                double * elements,
                double columnLower,
                double columnUpper, double objectiveValue)

        void setColumnLower(int whichColumn, double columnLower)
        void setColumnUpper(int whichColumn, double columnLower)
        void setRowLower(int whichColumn, double columnLower)
        void setRowUpper(int whichColumn, double columnLower)
        void setObjective(int whichColumn, double columnObjective)
        int numberRows()
        int numberColumns()


cdef class CyCoinModel:
    cdef CppCoinModel* CppSelf

#   cdef void CLP_addColumn(self, int numberInColumn,
#                                   int * rows,
#                                   double * elements,
#                                   double columnLower,
#                                   double columnUpper,
#                                   double objectiveValue,
#                                   char * name,
#                                   int isInteger)
#
    cdef void CLP_addColumn(self, int numberInColumn,
                            int * rows,
                            double * elements,
                            double columnLower,
                            double columnUpper,
                            double objectiveValue)

#   cdef void CLP_addRow(self, int numberInRow, int * columns,
#                       double * elements, double rowLower,
#                       double rowUpper, char * name)

    cdef void CLP_addRow(self, int numberInRow, int * columns,
                        double * elements, double rowLower,
                        double rowUpper)
