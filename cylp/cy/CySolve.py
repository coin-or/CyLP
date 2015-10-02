from __future__ import print_function
import sys
from cylp.cy.CyTest import CySolve

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Arg error: Usage example: python CySolve.py lp.mps d')
        sys.exit(0)
    method = sys.argv[2]
    if method == 'p':
        print('Positive edge is not implemented in Cython yet. Quitting.')
        sys.exit(1)
    CySolve(sys.argv[1], sys.argv[2])
