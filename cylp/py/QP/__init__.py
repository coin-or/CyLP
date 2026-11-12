import warnings

warnings.warn(
    'cylp.py.QP is deprecated and will be removed in CyLP 0.95. Its QP '
    'solvers no longer work. Clp solves convex QPs itself: read a QPS file '
    'with CyClpSimplex.readMps, or add the quadratic part of the objective '
    'with CyClpSimplex.loadQuadraticObjective, then call primal().',
    DeprecationWarning, stacklevel=2)
