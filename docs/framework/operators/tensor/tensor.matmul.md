# tensor.matmul

```rust 
   fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
```

Performs matrix product of two tensors.
The behavior depends on the dimensionality of the tensors as follows:
* If both tensors are 1-dimensional, the dot product is returned.
* If both arguments are 2-dimensional, the matrix-matrix product is returned.
* If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
* If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.

## Args

* `self`(`@Tensor<T>`) - the first tensor to be multiplied
* `other`(`@Tensor<T>`) - the second tensor to be multiplied

## Panics

* Panics if the dimension of the tensors is higher than two.

## Returns

A new `Tensor<T>` resulting from the matrix multiplication.

## Examples

Case 1: Dot product of two vectors (1D \* 1D)

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp16x16};

fn dot_product_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);

    let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);

    // We can call `matmul` function as follows.
    return tensor_1.matmul(@tensor_2);
}
>>> [5]
```

Case 2: Matrix multiplication (2D \* 2D)

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp16x16};

fn matrix_mul_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![244, 99, 109, 162].span()
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![151, 68, 121, 170].span()
    );

    // We can call `matmul` function as follows.
    return tensor_1.matmul(@tensor_2);
}
>>> [[48823, 33422],[36061, 34952]]
```

Case 3: Matrix-Vector multiplication (2D x 1D)

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp16x16};

fn matrix_vec_mul_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);

    // We can call `matmul` function as follows.
    return tensor_1.matmul(@tensor_2);
}
>>> [5,14,23]
```
