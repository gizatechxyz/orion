# tensor.group_normalization
 ``` rust 
fn group_normalization(self: @Tensor<T>, num_groups: usize, scale: @Tensor<T>, bias: @Tensor<T>, epsilon: Option<T>,) -> Tensor<T> ;
```
Computes group normalization of a given tensor. 
 The operator divides the channels into groups and computes the mean and variance for each group, followed by normalization over the group and across channels.

The overall process comprises of two stages:
1. The first stage normalizes the elements to have zero mean and unit variance for each instance.
2. The second stage scales and shifts the results of the first stage using the provided scale and bias tensors.
## Args

* `self` (`@Tensor<T>`) - The input tensor with dimensions `(N x C x D1 x D2 ... Dn)`, where `N` is the batch size,
  `C` is the number of channels, and `D1`, `D2`, ..., `Dn` are the remaining dimensions.
* `num_groups` (`usize`) - The number of groups to divide the channels into. It should be a divisor of the number of channels `C`.
* `scale` (`@Tensor<T>`) - Scale tensor of shape `(C)`.
* `bias` (`@Tensor<T>`) - Bias tensor of shape `(C)`.
* `epsilon` (`Option<T>`) (default is zero) - The epsilon value to use to avoid division by zero.

## Panics

* Panics if the input tensor's channels are not divisible by `num_groups`.
* Panics if the scale tensor's shape is not `(C)`.
* Panics if the bias tensor's shape is not `(C)`.

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
    data.append(FP16x16 { mag: 42237, sign: true });
    data.append(FP16x16 { mag: 17243, sign: true });
    data.append(FP16x16 { mag: 68912, sign: true });
    data.append(FP16x16 { mag: 97905, sign: true });
    data.append(FP16x16 { mag: 56532, sign: false });
    data.append(FP16x16 { mag: 52015, sign: true });
    data.append(FP16x16 { mag: 36893, sign: false });
    data.append(FP16x16 { mag: 73441, sign: true });
    data.append(FP16x16 { mag: 65232, sign: false });
    data.append(FP16x16 { mag: 26353, sign: false });
    data.append(FP16x16 { mag: 48427, sign: false });
    data.append(FP16x16 { mag: 177446, sign: true });
    data.append(FP16x16 { mag: 127569, sign: false });
    data.append(FP16x16 { mag: 111302, sign: false });
    data.append(FP16x16 { mag: 55209, sign: false });
    data.append(FP16x16 { mag: 39474, sign: true });
    data.append(FP16x16 { mag: 130149, sign: false });
    data.append(FP16x16 { mag: 16143, sign: false });
    data.append(FP16x16 { mag: 60665, sign: false });
    data.append(FP16x16 { mag: 36573, sign: true });
    data.append(FP16x16 { mag: 116495, sign: false });
    data.append(FP16x16 { mag: 38734, sign: true });
    data.append(FP16x16 { mag: 25428, sign: false });
    data.append(FP16x16 { mag: 86105, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 37043, sign: false });
    data.append(FP16x16 { mag: 54292, sign: false });
    data.append(FP16x16 { mag: 29626, sign: true });
    data.append(FP16x16 { mag: 44883, sign: true });
    let scale = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 280, sign: true });
    data.append(FP16x16 { mag: 77982, sign: true });
    data.append(FP16x16 { mag: 23584, sign: true });
    data.append(FP16x16 { mag: 9188, sign: false });
    let bias = TensorTrait::new(shape.span(), data.span());

    return X.group_normalization(2, @scale, @bias, epsilon); 
}
>>> [
     [[-16.5716552734375,3.568634033203125]
     [ 5.3177490234375, -0.3482513427734375]
     [ 4.0200653076171875, -4.3416595458984375]
     [ 21.036209106445312, 6.220306396484375]]

     [[ -7.13427734375,3.316986083984375,]
     [ 7.5245513916015625,-0.34033203125]
     [ -4.5717926025390625, -0.0133209228515625,]
     [ -2.1426239013671875, 16.505691528320312,]]

     [[ -6.1501922607421875, -19.385086059570312,]
     [ -0.0762786865234375, -4.130401611328125,]
     [ 5.2168731689453125,-1.3367462158203125,]
     [ 16.377029418945312, 35.31251525878906]]
     ]
``` 
