'''
This class defines an interface all the Python DUAL pivot rules must implement.
It consists of a method :py:func:`pivotRow` which must be implemented.
'''

from exceptions import NotImplementedError


class DualPivotPythonBase:
    def pivotRow(self):
        '''
        Every subclass of ``DualPivotPythonBase``, i.e. every pivot rule, must
        implement this method.

        Return the index of a selected row (constraint), an integer.
        '''
        raise NotImplementedError('pivotRow is not implemented')


