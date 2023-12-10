#tensor.pow

```rust
    fn pow(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
```

Pow takes input data (Tensor) and exponent Tensor, and produces one output data (Tensor) where the function f(x) = x^exponent, is applied to the data tensor elementwise.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor, base of the exponent.
* `other`(`@Tensor<T>`) - The second tensor, power of the exponent.

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<T>` with the same shape as the broadcasted inputs.

## Examples

Case 1: Compare tensors with same shape

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn pow_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 0, 1, 2, 0, 1, 2].span(),
    );

    return tensor_1.pow(@tensor_2);
}
>>> [0,1,4,0,4,25,0,7,64]
```

Case 2: Compare tensors with different shapes

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn pow_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![1, 3].span(), data: array![0, 1, 2].span(),
    );

    return tensor_1.pow(@tensor_2);
}
>>> [0,1,4,0,4,25,0,7,64]
```
