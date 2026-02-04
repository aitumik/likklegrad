import pytest
import numpy as np
from core.tensor import Tensor

def test_tensor_transpose_full():
    t = Tensor([[1, 2], [3, 4]])
    transposed_t = t.transpose()
    assert np.array_equal(transposed_t.data, np.array([[1, 3], [2, 4]]))

def test_tensor_transpose_dims():
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    transposed_t = t.transpose(0, 2)
    assert np.array_equal(transposed_t.data, np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]]))

def test_tensor_matmul():
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    result_t = t1.matmul(t2)
    expected_result = np.array([[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]])
    assert np.array_equal(result_t.data, expected_result)

def test_tensor_matmul_operator():
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    result_t = t1 @ t2
    expected_result = np.array([[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]])
    assert np.array_equal(result_t.data, expected_result)

def test_tensor_reshape():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    reshaped_t = t.reshape(3, 2)
    assert np.array_equal(reshaped_t.data, np.array([[1, 2], [3, 4], [5, 6]]))
    assert reshaped_t.shape == (3, 2)

def test_tensor_reshape_single_dim():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    reshaped_t = t.reshape(6)
    assert np.array_equal(reshaped_t.data, np.array([1, 2, 3, 4, 5, 6]))
    assert reshaped_t.shape == (6,)
