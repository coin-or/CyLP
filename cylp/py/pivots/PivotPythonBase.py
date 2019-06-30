'''
This class defines an interface all the Python pivot rules must implement.
It consists of a method :py:func:`pivotColumn` which must be implemented and
:py:func:`isPivotAcceptable` whose implementation is optional.
'''

class PivotPythonBase:
    def pivotColumn(self):
        '''
        Every subclass of ``PivotPythonBase``, i.e. every pivot rule, must
        implement this method.

        Return the index of the entering variable, an integer.
        '''
        raise NotImplementedError('pivotColumn is not implemented')

    def isPivotAcceptable(self):
        '''
        This is run just before the actual pivoting happens. Return False
        if the pivot is not acceptable for your method.
        '''
        return True

    def updateReducedCosts(self, updates, spareRow1, spareRow2, spareCol1, spareCol2):
        '''
        Update the reduced costs as its done in CLP's Dantzig rule.
        '''
        s = self.clpModel
        if updates.nElements:
            s.updateColumnTranspose(spareRow2, updates)
            s.transposeTimes(-1, updates, spareCol2, spareCol1)
            s.reducedCosts[s.nVariables:][updates.indices] -= updates.elements[:updates.nElements]
            s.reducedCosts[:s.nVariables][spareCol1.indices] -= spareCol1.elements[:spareCol1.nElements]
        updates.clear()
        spareCol1.clear()

