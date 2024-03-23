# TensorTrait::det

```rust
        fn det(tensor: @Tensor<T>) -> T;
```

Det calculates determinant of a square matrix or batches of square matrices. Det takes one input tensor of shape [*, M, M], where * is zero or more batch dimensions, and the inner-most 2 dimensions form square matrices. The output is a tensor of shape [*], containing the determinants of all input submatrices.

## Args

* `tensor`(`@Tensor<T>`) - The input tensor of shape [*, M, M].

## Returns

* The output is a tensor of shape [*]

## Examples

```rust
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;

fn example() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(2);
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    let input_0 = TensorTrait::new(shape.span(), data.span());

    return input_0.det();
}
>>> [0, -3]
```