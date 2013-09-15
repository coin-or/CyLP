.. CyLP Documentation documentation master file, created by
   sphinx-quickstart on Fri Nov  4 15:54:47 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. CyLP Documentation
.. ====================

.. CyLP is an interface to COIN-OR's Linear and mixed-integer program solvers
.. (CLP, CBC, and CGL). CyLP's unique feature is that you can use it to alter the
.. solution process of the solvers from within Python. For example, you may
.. define cut generators, branch-and-bound strategies, and primal/dual Simplex pivot
.. rules completely in Python.

.. You may use CyLP to model and solve linear programs (LP) of the form

.. .. math::
..     \mbox{minimize} \quad c^Tx \quad \mbox{subject to:} \ Ax = b, \ x \geq 0,

.. where :math:`c \in \mathbb{R}^n`, :math:`b \in \mathbb{R}^m`, :math:`A \in
.. \mathbb{R}^{m \times n}`. This process is controlled purely in Python while COIN-
.. OR's solvers are performing only behind the scenes.

.. You may read your LP from an ``mps`` file or use the CyLP's easy modeling
.. facility. Please find examples of how to do this in the documentation.


.. .. note::

..    CyLP interfaces a limited number of functionalities of
..    CLP. If there is any particular
..    class or method in CLP, CBC, and CGL that you need/want to use in Python
..    please don't hesitate to let us know; we will try to make the connections.
..    Moreover, in the case that you find a bug or a mistake, we would appreciate
..    it if you notify us. Contact us at mehdi [dot] towhidi [at] gerad [dot] ca.


.. toctree::
   :maxdepth: 2

.. include:: ../../../README.rst

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
