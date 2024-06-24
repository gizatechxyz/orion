import subprocess


def test_cairo():
    """
    Test function for the zk-backends cairo.
    """
    zk_backend = 'cairo'
    
    result = subprocess.run(['scarb', 'test'], cwd=f"zk-backends/{zk_backend}", capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)
    
    assert result.returncode == 0, f"Tests in {zk_backend} failed with return code {result.returncode}"
    