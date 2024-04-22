# TensorTrait::eye_like

```rust
     fn eye_like(self: @Tensor<T>, k: Option<i32>) -> Tensor<T>;
```

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the same as the input tensor. By default, the main diagonal is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals. 

## Args

* `self`(`@Tensor<T>`) - 2D input tensor to copy shape, and optionally, type information from.
* `k`(Option<i32>) - (Optional) Index of the diagonal to be populated with ones. Default is 0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, k > 0 populates an upper diagonal, and k < 0 populates a lower diagonal.

## Returns

* Output tensor, same shape as input tensor.

## Examples

```rust
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::numbers::{FixedTrait, FP8x23};


fn example() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: false });
    let tensor1 = TensorTrait::new(shape.span(), data.span());

    return tensor1.eye_like(Option::Some(0));
}
>>> [1, 0, 0, 1]
```
