Important Notice
==================
To comply with PEP8 we decided to rename the package name from CyLP to cylp,
which was long overdue.
It affects the package name ONLY and a simple replace can make your program
work with the new settings.
Thank you for your understanding.


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


Who uses CyLP
==============
CyLP is being used in a wide range of practical and research fields. Some of the users include:

#. PyArt, The Python ARM Radar Toolkit, used by Atmospheric Radiation Measurement
   (U.S. Department of energy). https://github.com/ARM-DOE/pyart
#. Meteorological Institute University of Bonn.
#. Sherbrooke university hospital (Centre hospitalier universitaire de Sherbrooke): CyLP is used for nurse scheduling.
#. Maisonneuve-Rosemont hospital (L'hôpital HMR): CyLP is used for  physician scheduling with preferences.
#. Lehigh University: CyLP is used to teach mixed-integer cuts.
#. IBM T. J. Watson research center
#. Saarland University, Germany



Installation
============

The easiest way to install CyLP is by using the binaries. If that's not
possible you may always compile it from source.

Requirements
--------------

CyLP needs Numpy (www.numpy.org) and Scipy (www.scipy.org). If you wish to install CyLP from source, you will also need to compile Cbc. Details of this process is given below.

Binary Installation
----------------------

If you have setuptools installed you may run::

    $ easy_install cylp

If a binary is available for your architecture it will be installed. Otherwise
you will see an error telling you to specify where to find a Cbc installation.
That's because easy_install is trying to compile the source. In this case
you'll have to compile Cbc and set and environment variable to point to it
before calling easy_install again. The details are given in the `Installing
from source`_ section.



Installing from source
========================

STEP 1:
    Install CBC (http://www.coin-or.org/download/source/Cbc/).
    CyLP can be compiled against
    Cbc version 2.8.5. Please go to the installation directory and run::

        $ ./configure
        $ make
        $ make install

STEP 2:
    Create an environment variable called COIN_INSTALL_DIR pointing to your
    installation of Coin. For example::

        $ export COIN_INSTALL_DIR=/Users/mehdi/Cbc-2.8.5

You may also add this line to your ~/.bash_rc or ~/.profile to make
it persistent.

STEP 3:
    Install CyLP. Go to CyLP's root directory and run::

        $ python setup.py install

STEP 4 (LINUX):
     In linux you might also need to add COIN's lib directory to
     LD_LIBRARY_PATH as follows::

        $ export LD_LIBRARY_PATH=/path/to/Cbc-2.8.5/lib:$LD_LIBRARY_PATH"

Optional step:
    If you want to run the doctests (i.e. ``make doctest`` in the ``doc`` directory)
    you should also define::

        $ export CYLP_SOURCE_DIR=/Path/to/cylp

Now you can use CyLP in your python code. For example::

    >>> from cylp.cy import CyClpSimplex
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
    from cylp.cy import CyClpSimplex
    from cylp.py.modeling.CyLPModel import CyLPArray

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

This is the expected output::

    Clp0006I 0  Obj 1.1 Primal inf 2.8999998 (2) Dual inf 5.01e+10 (5) w.o. free dual inf (4)
    Clp0006I 5  Obj 1.3
    Clp0000I Optimal - objective value 1.3
    [ 0.2  2.   1.1]

Documentation
===============
You may access CyLP's documentation:

1. *Online* : Please visit http://mpy.github.io/CyLPdoc/

2. *Offline* : To install CyLP's documentation in your repository, you need
   Sphinx (http://sphinx.pocoo.org/). You can generate the documentation by
   going to cylp/doc and run ``make html`` or ``make latex`` and access the
   documentation under cylp/doc/build. You can also run ``make doctest`` to
   perform all the doctest.


.. image:: https://d2weczhvl823v0.cloudfront.net/mpy/cylp/trend.png
   :alt: Bitdeli badge
   :target: https://bitdeli.com/free

.. image:: https://cruel-carlota.pagodabox.com/f8efbddd4f44bb098d20dafdd0b9e897
   :alt: githalytics.com
   :target: http://githalytics.com/mpy/cylp
