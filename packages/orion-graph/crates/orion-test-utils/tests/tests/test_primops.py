import numpy as np
from utils import run_test

#Tests PrimOps
def test_sin():
    x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float64).reshape(2, 2)
    run_test('test_sin', {'x': x})
    
def test_add():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    run_test('test_add', {'a': a, 'b': b})
    
def test_add_broadcast():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([1, 2], dtype=np.float64)
    run_test('test_add_broadcast', {'a': a, 'b': b})
    
def test_add_broadcast_scalar():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([2], dtype=np.float64)
    run_test('test_add_broadcast_scalar', {'a': a, 'b': b})

def test_mul():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([2, 2, 4, 4], dtype=np.float64).reshape(2, 2)
    run_test('test_mul', {'a': a, 'b': b})
    
def test_mul_broadcast():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([1, 2], dtype=np.float64)
    run_test('test_mul_broadcast', {'a': a, 'b': b})
    
def test_mul_broadcast_scalar():
    a = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
    b = np.array([2], dtype=np.float64)
    run_test('test_mul_broadcast_scalar', {'a': a, 'b': b})
    
def test_log():
    x = np.array([1, 2, 4, 8], dtype=np.float64).reshape(2, 2)
    run_test('test_log', {'x': x})
    
def test_exp():
    x = np.array([0, 1, 2, 3], dtype=np.float64).reshape(2, 2)
    run_test('test_exp', {'x': x})
    
def test_sqrt():
    x = np.array([0, 1, 4, 9], dtype=np.float64).reshape(2, 2)
    run_test('test_sqrt', {'x': x})
    
def test_recip():
    x = np.array([0.5, 1, 4, 9], dtype=np.float64).reshape(2, 2)
    run_test('test_recip', {'x': x})
    
def test_mod():
    a = np.array([7, 10], dtype=np.float64).reshape(2, 1)
    b = np.array([3, 4, 5], dtype=np.float64).reshape(1, 3)
    run_test('test_mod', {'a': a, 'b': b})
    
def test_less():
    a = np.array([7, 10], dtype=np.float64).reshape(2, 1)
    b = np.array([3, 40, 5], dtype=np.float64).reshape(1, 3)
    run_test('test_less', {'a': a, 'b': b})
    
def test_reducesum_keepdim():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(3, 2, 2)
    run_test('test_reducesum_keepdim', {'x': x})
    
def test_reducesum_not_keepdim():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(3, 2, 2)
    run_test('test_reducesum_not_keepdim', {'x': x})
    
def test_reducemax_keepdim():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(3, 2, 2)
    run_test('test_reducemax_keepdim', {'x': x})
    
def test_reducemax_not_keepdim():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(3, 2, 2)
    run_test('test_reducemax_not_keepdim', {'x': x})

