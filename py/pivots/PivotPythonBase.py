'''
This class defines an interface all the Python pivot rules must implement.
It consists of a method :py:func:`pivotColumn` which must be implemented and
:py:func:`isPivotAcceptable` whose implementation is optional.
'''

from exceptions import NotImplementedError


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
