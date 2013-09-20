What is CyLP?
==============
CyLP is a Python interface to COIN-OR’s Linear and mixed-integer program solvers
(CLP, CBC, and CGL). CyLP’s unique feature is that you can use it to alter the
solution process of the solvers from within Python. For example, you may
define cut generators, branch-and-bound strategies, and primal/dual Simplex
pivot rules completely in Python.

You may read your LP from an mps file or use the CyLP’s easy modeling
facility. Please find examples in the `documentation
<http://mpy.github.io/CyLPdoc/>`_.

.. note::

   CyLP interfaces a limited number of functionalities of
   COIN-OR’s solvers. If there is any particular
   class or method in CLP, CBC, and CGL that you would like to use in Python
   please don't hesitate to let us know; we will try to make the connections.
   Moreover, in the case that you find a bug or a mistake, we would appreciate
   it if you notify us. Contact us at mehdi [dot] towhidi [at] gerad [dot] ca.




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

Now you can use CyLP in your python code. For example::

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



Modeling Example
==================

Here is an example of how to model with CyLP's modeling facility::

    import numpy as np
    from CyLP.cy import CyClpSimplex
    from CyLP.py.modeling.CyLPModel import CyLPArray

    s = CyClpSimplex()

    # Add variables
    x = s.addVariable('x', 3)
    y = s.addVariable('y', 2)

    # Create coefficients and bounds
    A = np.matrix([[1., 2., 0],[1., 0, 1.]])
    B = np.matrix([[1., 0, 0], [0, 0, 1.]])
    D = np.matrix([[1., 2.],[0, 1]])
    a = CyLPArray([5, 2.5])
    b = CyLPArray([4.2, 3])
    x_u= CyLPArray([2., 3.5])

    # Add constraints
    s += A * x <= a
    s += 2 <= B * x + D * y <= b
    s += y >= 0
    s += 1.1 <= x[1:3] <= x_u

    # Set the objective function
    c = CyLPArray([1., -2., 3.])
    s.objective = c * x + 2 * y.sum()

    # Solve using primal Simplex
    s.primal()
    print s.primalVariableSolution['x']



Documentation
===============
You may access CyLP's documentation:

1. *Online* : Please visit http://mpy.github.io/CyLPdoc/

2. *Offline* : To install CyLP's documentation in your repository, you need
   Sphinx (http://sphinx.pocoo.org/). You can generate the documentation by
   going to CyLP/doc and run ``make html`` or ``make latex`` and access the
   documentation under CyLP/doc/build. You can also run ``make doctest`` to
   perform all the doctest.


.. image:: https://d2weczhvl823v0.cloudfront.net/mpy/cylp/trend.png
   :alt: Bitdeli badge
   :target: https://bitdeli.com/free

