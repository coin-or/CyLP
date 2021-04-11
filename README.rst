CyLP
====

CyLP is a Python interface to COIN-OR’s Linear and mixed-integer program solvers
(CLP, CBC, and CGL). CyLP’s unique feature is that you can use it to alter the
solution process of the solvers from within Python. For example, you may
define cut generators, branch-and-bound strategies, and primal/dual Simplex
pivot rules completely in Python.

You may read your LP from an mps file or use the CyLP’s easy modeling
facility. Please find examples in the `documentation
<http://coin-or.github.io/CyLP/>`_.

Docker
======

If you're comfortable with Docker, you can get started right away with the container 
available on Dockerhub that comes with CyLP pre-installed. 

https://hub.docker.com/repository/docker/coinor/cylp

Otherwise, read on. 

Prerequisites
=============

CyLP depends on Numpy (www.numpy.org) and Scipy (www.scipy.org). Please note that 
Numpy does need to be installed prior to installing CyLP,
even though it is listed as a dependency in the `setup.py` file.

You will also need to install binaries for Cbc. The version should be 2.10 or earlier 
(current master branch of Cbc will not work with this version of CyLP).
You can install Cbc by either by 
installing with a package manager, by downloading pre-built binaries,
or by building yourself from source using `coinbrew <https://github.com/coin-or/coinbrew>`_.

    1. To install Cbc in Linux, the easiest way is to use a package manager. Install `coinor-libcbc-dev` on Ubuntu/Debian 
    or `coin-or-Cbc-devel` on Fedora. Cbc is also available on Linux through conda with  

       ``$ conda create -n cbc coin-or-cbc -c conda-forge``
       
    2. On OS X, it is easiest to install Cbc with homebrew:
    
       ``$ brew tap coin-or-tools/coinor``
        
       ``$ brew install coin-or-tools/coinor/cbc pkg-config``

    Cbc is also available on OS X through conda with  

       ``$ conda create -n cbc coin-or-cbc -c conda-forge``  

    3. On Windows, a binary wheel is available and it is not necessary to install Cbc.
    
You should no longer need to build Cbc from source on any platform unless for some reason, none of the
above recipes applies to you. If you do need to build from source, please go to the `Cbc <https://github.com/coin-or/Cbc>`_
project page and follow the instructions there. After building and installing, make sure to 
either set the `COIN_INSTALL_DIR` variable to point to the installation or set `PKG_CONFIG_PATH` to point to
the directory where the `.pc` files are installed. You may also need to set either `LD_LIBRARY_PATH` (Linux)
or `DYLD_LIBRARY_PATH` (OS X).

Installation
============

Once Numpy and Cbc are installed, simply do::

    $ pip install cylp

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

1. *Online* : Please visit http://coin-or.github.io/CyLP/

2. *Offline* : To install CyLP's documentation in your repository, you need
   Sphinx (http://sphinx-doc.org/). You can generate the documentation by
   going to cylp/doc and run ``make html`` or ``make latex`` and access the
   documentation under cylp/doc/build. You can also run ``make doctest`` to
   perform all the doctest.
   
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


