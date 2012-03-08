.. CyLP Documentation documentation master file, created by
   sphinx-quickstart on Fri Nov  4 15:54:47 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CyLP Documentation
==============================================

Consider a linear program (LP) of the form

.. math::
    \mbox{minimize} \quad c^Tx \quad \mbox{subject to:} \ Ax = b, \ x \geq 0,
    
where :math:`c \in \mathbb{R}^n`, :math:`b \in \mathbb{R}^m`, :math:`A \in
\mathbb{R}^{m \times n}`. You can solve an LP using COIN-OR's CLP
(https://projects.coin-or.org/Clp)\ ---an opensource project written in C++.
Furthermore, if you demand some variables to be integer, then the problem is
called a mixed-integer program (MIP). CBC(https://projects.coin-or.org/Cbc) 
is used in combination with
CGL(https://projects.coin-or.org/Cgl) to solve MIPs.

CyLP is a Cython interface to COIN-OR's CLP, CBC, and CGL. At its most basic it
provides a Python API to CLP. CyLP interfaces CLP to read an LP from an
``mps`` file and solve it. In addition, CyLP provides a modeling facility
which allows easy creation of LPs. 

But what makes CyLP completely different is its capability to alter the
solution process. For example it provides a framework to define simplex pivot
rules in Python. In the MIP part, it allows definition of branch-and-cut tree
transversal rule in Python.  

This documentation is incomplete but we, to the best of our ability, are trying
to improve it. 


.. note::

   CyLP interfaces a very limited number of functionalities of
   CLP---those that became necessary along the way. If there is any particular
   class or method in CLP, CBC, and CGL that you need/want to use in Python
   please don't hesitate to let us know; we will try to make the connections.
   Moreover, in the case that you find a bug or a mistake, we would appriciate
   it if you notify us. Contact us at mehdi [dot] towhidi [at] gerad [dot] ca. 


.. toctree::
   :maxdepth: 2

.. include:: ../../README.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Modules
==========================

.. toctree::
   :maxdepth: 3 
   
   modules/cy
   modules/py
