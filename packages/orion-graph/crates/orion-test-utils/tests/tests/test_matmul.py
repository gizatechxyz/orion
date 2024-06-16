import numpy as np
from utils import run_test

# Tests MatMul 
def test_matmul_ax_b():
    A = np.array([1, 2, 3, 4], dtype=np.float64)
    B = np.array([1, 2, 3, 4], dtype=np.float64)
    run_test('test_matmul_ax_a', {'A': A, 'B': B})
      
def test_matmul_abx_b():
    A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64).reshape(3, 4)
    B = np.array([1, 2, 3, 4], dtype=np.float64)
    run_test('test_matmul_abx_b', {'A': A, 'B': B})
      
def test_matmul_ax_ab():
    A = np.array([1, 2, 3], dtype=np.float64)
    B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64).reshape(3, 4)
    run_test('test_matmul_ax_ab', {'A': A, 'B': B})
    
def test_matmul_abcx_cd():
    A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(3, 2, 2)
    B = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(2, 6)
    run_test('test_matmul_abcx_cd', {'A': A, 'B': B})
    
def test_matmul_abcx_acd():
    A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(2, 3, 4)
    B = np.array([4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0], dtype=np.float64).reshape(2, 4, 3)
    run_test('test_matmul_abcx_acd', {'A': A, 'B': B})
 
def test_matmul_abcdx_abde():
    A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(2, 2, 3, 2)
    B = np.array([4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 40.0, 40.0, 40.0, 60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0], dtype=np.float64).reshape(2, 2, 2, 3)
    run_test('test_matmul_abcdx_abde', {'A': A, 'B': B})
    
def test_matmul_abcdex_abcef():
    A = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0], dtype=np.float64).reshape(2, 2, 2, 3, 2)
    B = np.array([24, 60, 69, 88, 3, 6, 21, 47, 30, 18, 53, 73, 9, 38, 77, 89, 61, 14, 39, 33, 36, 19, 50, 53, 10, 95, 44, 57, 66, 98, 95, 37, 93, 47, 49, 80, 5, 85, 58, 0, 1, 10, 86, 72, 3, 6, 82, 65], dtype=np.float64).reshape(2, 2, 2, 2, 3)
    run_test('test_matmul_abcdex_abcef', {'A': A, 'B': B})

# Tests matrices 2x2   
def test_matmul():
    A = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    B = [[9, 8, 7],[6, 5, 4],[3, 2, 1]]
    run_test('test_matmul', {'A': A, 'B': B})

def test_matmul_identity_matrix():
    A = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
    B = [[9, 8, 7],[6, 5, 4],[3, 2, 1]]
    run_test('test_matmul_identity_matrix', {'A': A, 'B': B})

def test_matmul_zero_matrix():
    A = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    B = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    run_test('test_matmul_zero_matrix', {'A': A, 'B': B})

def test_matmul_non_square_matrix():
    A = [[1, 2, 3],[4, 5, 6]]
    B = [[7, 8],[9, 10],[11, 12]]
    run_test('test_matmul_non_square_matrix', {'A': A, 'B': B})

# Tests MatMul initializer : one input is a graph input, the other is an initializer
def test_matmul_initializer_A():
    B = [[7, 8, 9], [10, 11, 12]]
    run_test('test_matmul_initializer_A', {'B': B})
     
def test_matmul_initializer_B():
    A = [[7, 8],[9, 10],[11, 12]]
    run_test('test_matmul_initializer_B', {'A': A})

# Tests MatMul multi node : inputs coming from two different nodes 
def test_matmul_multi_nodes():
    A = [[1, 2, 3],[4, 5, 6]]
    B = [[7, 8],[9, 10],[11, 12]]
    run_test('test_matmul_multi_nodes', {'A': A, 'B': B})