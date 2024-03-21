# tensor.group_normalization
 ``` rust 
fn group_normalization(self: @Tensor<T>, num_groups: usize, scale: @Tensor<T>, bias: @Tensor<T>, epsilon: Option<T>,) -> Tensor<T> ;
```
Computes group normalization of the input tensor.

The overall computation has two stages:
1. The first stage normalizes the elements to have zero mean and unit variance for each instance in each group.
2. The second stage scales and shifts the results of the first stage using the provided scale and bias tensors.

## Args

* `self` (`@Tensor<T>`) - The input tensor with dimensions `(N x C x D1 x D2 ... Dn)`, where `N` is the batch size,
  `C` is the number of channels, and `D1`, `D2`, ..., `Dn` are the remaining dimensions.
* `num_groups` (`usize`) - The number of groups to divide the channels into. `num_groups` needs to be divisible by the number of channels `C` present.
* `scale` (`@Tensor<T>`) - Scale tensor of shape `(C)`.
* `bias` (`@Tensor<T>`) - Bias tensor of shape `(C)`.
* `epsilon` (`Option<T>`) (default is zero) - The epsilon value to use to avoid division by zero.

## Panics

* Panics if the input tensor's channels are not divisible by `num_groups`.

## Returns

A new tensor `Tensor<T>` with the same shape as the input tensor, after applying group normalization.

## Examples

```rust
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn group_normalization_example() -> Tensor<FP16x16> {
   
    let epsilon = Option::Some( FixedTrait::new(6554, false));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24149, sign: false });
    data.append(FP16x16 { mag: 33227, sign: true });
    data.append(FP16x16 { mag: 6482, sign: false });
    data.append(FP16x16 { mag: 70109, sign: false });
    data.append(FP16x16 { mag: 25079, sign: true });
    data.append(FP16x16 { mag: 69218, sign: true });
    data.append(FP16x16 { mag: 29877, sign: false });
    data.append(FP16x16 { mag: 3983, sign: true });
    data.append(FP16x16 { mag: 53517, sign: true });
    data.append(FP16x16 { mag: 68285, sign: true });
    data.append(FP16x16 { mag: 78369, sign: true });
    data.append(FP16x16 { mag: 49571, sign: false });
    data.append(FP16x16 { mag: 60527, sign: true });
    data.append(FP16x16 { mag: 35352, sign: false });
    data.append(FP16x16 { mag: 33514, sign: true });
    data.append(FP16x16 { mag: 7683, sign: false });
    data.append(FP16x16 { mag: 14398, sign: true });
    data.append(FP16x16 { mag: 28678, sign: true });
    data.append(FP16x16 { mag: 52124, sign: false });
    data.append(FP16x16 { mag: 7433, sign: true });
    data.append(FP16x16 { mag: 157814, sign: false });
    data.append(FP16x16 { mag: 47280, sign: false });
    data.append(FP16x16 { mag: 24880, sign: true });
    data.append(FP16x16 { mag: 138131, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 47324, sign: true });
    data.append(FP16x16 { mag: 39055, sign: true });
    data.append(FP16x16 { mag: 85442, sign: true });
    data.append(FP16x16 { mag: 74074, sign: true });
    let scale = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 68434, sign: false });
    data.append(FP16x16 { mag: 112102, sign: false });
    data.append(FP16x16 { mag: 8495, sign: true });
    data.append(FP16x16 { mag: 73947, sign: true });
    let bias = TensorTrait::new(shape.span(), data.span());


    return X.group_normalization(2,scale,bias, epsilon); 
}
>>>
[
 [[0.92068481, 1.8956604 ],
  [1.85632324, 0.96403503],
  [0.12124634, 1.50914001],
  [2.40892029, 1.48588562]],

 [[1.25189209, 1.44520569],
  [2.15039062, 0.76837158],
  [1.34228516, 1.61160278],
  [0.57377625, 1.67410278]],

 [[1.33195496, 1.60957336],
  [0.88078308, 1.83625793],
  [1.46817017, 0.42315674],
  [0.42134094, 1.99681091]]]
``` 
