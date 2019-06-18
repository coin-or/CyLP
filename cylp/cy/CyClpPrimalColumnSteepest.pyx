# cython: embedsignature=True

from __future__ import print_function


#cimport cClpPrimalColumnSteepest
from cClpPrimalColumnSteepest cimport c_ClpPrimalColumnSteepest, new_ClpPrimalColumnSteepest
#cimport cClpPrimalColumnPivot
#from cClpPrimalColumnPivot cimport ClpPrimalColumnPivot
from CyClpPrimalColumnPivot import ClpPrimalColumnPivot
cdef class ClpPrimalColumnSteepest(ClpPrimalColumnPivot):
	cdef c_ClpPrimalColumnSteepest *thisptr      # hold a C++ instance which we're wrapping
	def __cinit__(self, mode):
		self.thisptr = new_ClpPrimalColumnSteepest(mode)
	
		
#cdef Rectangle r = Rectangle(1, 2, 3, 4)
#print(r)
#print("Original area:", r.getArea())
#r.move(1,2)
#print(r)
#print("Area is invariante under rigid motions:", r.getArea())
#r += Rectangle(0,0,1,1)
#print(r)
#print("Now the aread is:", r.getArea())

