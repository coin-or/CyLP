Installation
============

STEP 1: 
    Install CoinMP. You can get the source at
    http://www.coin-or.org/download/source/CoinMP/. CyLP can be compiled against
    CoinMP-1.6.0 or CoinMP-1.4.0 (earlier versions might work too). If you
    choose CoinMP 1.6.0, to compile CyLP you will need a LAPACK
    implementation and BZIP2 installed. If you are on a MAC or a linux
    system you might already have both. CoinMP-1.4.0 does not have this
    requirement.  
    To compile CoinMP's source you may need to pass 'g95' to configure::

        $ ./configure F77=/path/to/g95
        $ make
        $ make install

STEP 2: 
    Edit setup.py and set CoinDir to the location of the top directory of 
    CoinMP.

STEP 3: 
    Install CyLP. Go to CyLP's root directory and run 'make'

STEP 4: 
    Update you PYTHONPATH environment variable to contain the 
    parent directory of CyLP. For example, if the path to CyLP 
    installation is /Users/Mehdi/CyLP, then '/Users/Mehdi' must
    be in your PYTHONPATH. To this end, you may run::
 
        $ export PYTHONPATH="/Users/Mehdi/:$PYTHONPATH"


Now you can use CyLP in your python code. For example:
    
    >>> from CyLP.cy import CyClpSimplex
    >>> s = CyClpSimplex()
    >>> s.readMps('../input/netlib/adlittle.mps')
    0
    >>> s.initialSolve()
    'optimal'
    >>> round(s.objectiveValue, 3)
    225494.963
    
Or simply go to CyLP and run:: 

    $ python -m unittest discover

to run all CyLP unit tests.



Usage
=====

To run the primal Simplex method on a problem in MPS format, use::

    $ python python/PySolve.py input/netlib/25fv47.mps d

or::

    $ python cython/CySolve.py input/netlib/25fv47.mps d

Use `p` instead of the trailing `d` to use the positive edge pivot rule instead of Dantzig's canonical pivot rule.


Documentation
===============
You may access CyLP's documentation:

    1. *Online* : http://mpy.github.com/CyLP.

    2. *Offline* : To install CyLP's documentation in your repository, you need Sphinx (http://sphinx.pocoo.org/). You can generate the documentation by going to CyLP/doc and run ``make html`` or ``make latex`` and access the documentation under CyLP/doc/build. You can also run ``make doctest`` to perform all the doctest. 
