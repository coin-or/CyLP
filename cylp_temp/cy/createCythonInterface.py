#!/usr/bin/env python

# Usage:
# The command:
# > python createCythonInterface.py Person
# creates the appropriate Cython pyx and pxd files


import sys


className = sys.argv[1]
cyclassname = 'Cy' + className
pyxFile = cyclassname + '.pyx'
pxdFile = cyclassname + '.pxd'

# First create the pxd file:
pxdContent = \
'''
cdef extern from "%s.hpp":
    cdef cppclass Cpp%s "%s":
        pass
    Cpp%s *new_Cpp%s "new %s" ()

cdef class Cy%s:
    cdef Cpp%s* CppSelf
    cdef setCppSelf(self, Cpp%s* s)
''' % (className, className, className, className, className,
            className, className, className, className)

with open(pxdFile, 'w') as f:
    f.write(pxdContent)


#Create the pyx file
pyxContent = '''
cimport Cy%s

cdef class Cy%s:
    'Cy%s documentation'
    def __cinit__(self):
        self.CppSelf = new Cpp%s()

    cdef setCppSelf(self, Cpp%s* s):
        del self.CppSelf
        self.CppSelf = s

''' % (className, className, className, className, className)

with open(pyxFile, 'w') as f:
    f.write(pyxContent)


