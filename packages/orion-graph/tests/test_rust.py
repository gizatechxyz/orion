import subprocess
import pytest

def run_cargo_test(crate):
    """
    Runs cargo test on a specified crate and returns the subprocess result.
    """
    return subprocess.run(['cargo', 'test'], cwd=f"crates/{crate}", capture_output=True, text=True)

def check_test_results(result, crate):
    """
    Prints the stdout and stderr of the test result and checks the return code.
    Asserts that the test passed by checking the return code is zero.
    """
    print(result.stdout)
    print(result.stderr)
    
    assert result.returncode == 0, f"Tests in {crate} failed with return code {result.returncode}"

def test_orion_test_utils():
    """
    Test function for the orion-test-utils crate.
    """
    result = run_cargo_test('orion-test-utils')
    check_test_results(result, 'orion-test-utils')

def test_orion():
    """
    Test function for the orion crate.
    """
    result = run_cargo_test('orion')
    check_test_results(result, 'orion')

def test_primops_decomp():
    """
    Test function for the primops-decomp crate.
    """
    result = run_cargo_test('primops-decomp')
    check_test_results(result, 'primops-decomp')

@pytest.mark.parametrize("test_func", [
    test_orion_test_utils,
    test_orion,
    test_primops_decomp
])
def test_crate_tests(test_func):
    """
    Pytest function to test different crates.
    """
    test_func()
