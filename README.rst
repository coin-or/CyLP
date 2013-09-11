Installation
============

STEP 1:
    Install CoinMP. You can get the source at
    http://www.coin-or.org/download/source/CoinMP/. CyLP can be compiled against
    CoinMP-1.6.0. To compile CyLP you will need a LAPACK
    implementation and BZIP2 installed. If you are on a MAC or a linux
    system you might already have both.
    To compile CoinMP's source you may need to pass 'g95' to configure::

        $ ./configure F77=/path/to/g95
        $ make
        $ make install

STEP 2:
    Create an environment variable called COIN_INSTALL_DIR pointing to your
    installation of Coin. For example::

        $ export COIN_INSTALL_DIR=/Users/mehdi/CoinMP-1.6.0

    You may also add this line to your ~/.bash_rc or ~/.profile to make
    it persistent.

STEP 3:
    Install CyLP. Go to CyLP's root directory and run::

        $ python setup.py install

STEP 4 (LINUX):
     In linux you might also need to add COIN's lib directory to
     LD_LIBRARY_PATH as follows::

        $ export LD_LIBRARY_PATH=/path/to/CoinMP-1.6.0/lib:$LD_LIBRARY_PATH"

Optional step:
    If you want to run the doctests (i.e. ``make doctest`` in the ``doc`` directory)
    you should also define::

        $ export CYLP_SOURCE_DIR=/Path/to/CyLP

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
=======

To run the primal Simplex method on a problem in MPS format, use::

    $ python CyLP/py/PySolve.py input/netlib/25fv47.mps d

or::

    $ python CyLP/cy/CySolve.py input/netlib/25fv47.mps d

Use `p` instead of the trailing `d` to use the positive edge pivot rule instead of Dantzig's canonical pivot rule.


Documentation
===============
You may access CyLP's documentation:

    1. *Online* : http://mpy.github.com/CyLP.

    2. *Offline* : To install CyLP's documentation in your repository, you need Sphinx (http://sphinx.pocoo.org/). You can generate the documentation by going to CyLP/doc and run ``make html`` or ``make latex`` and access the documentation under CyLP/doc/build. You can also run ``make doctest`` to perform all the doctest.
