"""
run_all_tests.py
Test runner script to execute all automated test suites and report results.
"""

import sys
import unittest

def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover and add all test files
    discovered = loader.discover(".", pattern="test_*.py")
    suite.addTests(discovered)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("TEST SUITE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
