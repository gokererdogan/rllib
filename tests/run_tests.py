"""
rllib - Reinforcement Learning Library

Script for running tests. Enables the user to skip slow tests and run only the selected test modules.
    Run using
        python run_tests.py module1 module2 ... --skipslow

To run coverage analysis run (requires coverage.py to be installed)
    coverage run --source ../ run_tests.py module1 module 2 ... --skipslow

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run rllib unittests.")
    parser.add_argument('modules', type=str, nargs='+', help='Test module names to run. If discover, '
                                                             'uses unittest.discover to find tests in '
                                                             'the current folder.')
    # note that --skipslow parameter seems to have no effect here but it is checked in TestCase classes using
    # unittest skipIf decorator.
    parser.add_argument('--skipslow', action='store_true', help='Do not run slow tests.')

    args = parser.parse_args()

    loader = unittest.TestLoader()
    if 'discover' in args.modules:
        tests = loader.discover('./')
    else:
        tests = loader.loadTestsFromNames(args.modules)

    unittest.TextTestRunner(verbosity=2).run(tests)
