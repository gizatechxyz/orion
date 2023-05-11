# tensor.matmul

Performs matrix product of two tensors.

```rust
fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
```

The behavior depends on the dimensionality of the tensors as follows:

* If both tensors are 1-dimensional, the dot product is returned.
* If both arguments are 2-dimensional, the matrix-matrix product is returned.
* If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
* If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.

#### Args

| Name    | Type         | Description                        |
| ------- | ------------ | ---------------------------------- |
| `self`  | `@Tensor<T>` | the first tensor to be multiplied  |
| `other` | `@Tensor<T>` | the second tensor to be multiplied |

> _`<T>` generic type depends on Tensor dtype._

#### Panics

| TypeError                                                  |
| ---------------------------------------------------------- |
| Panics if the dimension of the tensors is higher than two. |

#### Returns

A new `Tensor<T>` resulting from the matrix multiplication.

> _`<T>` generic type depends on Tensor dtype._

#### Examples

Case 1: Dot product of two vectors (1D \* 1D)

```rust
fn dot_product_example() -> Tensor<u32> {
  // We instantiate two 1D Tensor here.
  // tensor_1 = [0,1,2]
  // tensor_2 = [0,1,2]
  let tensor_1 = u32_tensor_1x3_helper();
  let tensor_2 = u32_tensor_1x3_helper();		
		
  // We can call `matmul` function as follows.
  return tensor_1.matmul(@tensor_2);
}
>>> [5]
```

Case 2: Matrix multiplication (2D \* 2D)

```rust
fn matrix_mul_example() -> Tensor<u32> {
    // We instantiate two 2D Tensor here.
    // tensor_1 = [0,1,2]
    // tensor_2 = [0,1,2]
    let tensor_1 = u32_tensor_2x2_helper();		
    let tensor_2 = u32_tensor_2x2_helper();

    // We can call `matmul` function as follows.
    return tensor_1.matmul(@tensor_2);
}
>>> [[2,3],[6,11]]
```

Case 3: Matrix-Vector multiplication (2D x 1D)

```rust
fn matrix_vec_mul_example() -> Tensor<u32> {
    // We instantiate two 2D Tensor here.
    // tensor_1 = [[0,1,2],[3,4,5],[6,7,8]]
    // tensor_2 = [0,1,2]
    let tensor_1 = u32_tensor_3x3_helper();
    let tensor_2 = u32_tensor_1x3_helper();
		
    // We can call `matmul` function as follows.
    return tensor_1.matmul(@tensor_2);
}
>>> [5,14,23]
```
