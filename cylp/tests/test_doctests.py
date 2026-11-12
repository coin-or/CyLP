import unittest
import doctest
import importlib
import numpy as np

# The modules with examples in their docstrings
modules = ['cylp.cy.CyCbcModel', 'cylp.cy.CyClpSimplex',
           'cylp.cy.CyCoinIndexedVector', 'cylp.cy.CyCoinModel',
           'cylp.cy.CyCoinMpsIO', 'cylp.cy.CyCoinPackedMatrix',
           'cylp.py.modeling.CyLPModel', 'cylp.py.pivots.DantzigPivot',
           'cylp.py.pivots.DualDantzigPivot', 'cylp.py.pivots.LIFOPivot',
           'cylp.py.pivots.MostFrequentPivot',
           'cylp.py.pivots.PositiveEdgePivot', 'cylp.py.utils.sparseUtil']


class TestDocstrings(unittest.TestCase):

    def test_examples(self):
        # The examples show NumPy scalars the way NumPy 1 prints them,
        # e.g. True rather than np.True_
        legacy = '1.25' if np.lib.NumpyVersion(np.__version__) >= '2.0.0' \
                 else False
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                runner = doctest.DocTestRunner(
                    optionflags=doctest.ELLIPSIS |
                                doctest.NORMALIZE_WHITESPACE)
                report = []
                done = set()
                with np.printoptions(legacy=legacy):
                    for test in doctest.DocTestFinder().find(module, name):
                        # Cython also lists the docstrings of a module in
                        # its __test__ dictionary, so some are found twice.
                        # The copies can differ in the signature that
                        # embedsignature adds, but not in their examples.
                        examples = tuple((e.source, e.want)
                                         for e in test.examples)
                        if examples in done:
                            continue
                        done.add(examples)
                        runner.run(test, out=report.append)
                results = runner.summarize(verbose=False)
                self.assertGreater(results.attempted, 0)
                self.assertEqual(results.failed, 0, ''.join(report))


if __name__ == '__main__':
    unittest.main()
