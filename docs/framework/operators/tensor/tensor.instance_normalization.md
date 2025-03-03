# tensor.instance_normalization

```rust
   fn instance_normalization(
    self: @Tensor<T>,
    scale: @Tensor<T>,
    bias: @Tensor<T>,
    epsilon: Option<T>,
) -> Tensor<T>;
```

Computes instance normalization on a given input tensor.

The overall computation has two stages:
1. The first stage normalizes the elements to have zero mean and unit variance for each instance.
2. The second stage scales and shifts the results of the first stage using the provided scale and bias tensors.

## Args

* `self` (`@Tensor<T>`) - The input tensor with dimensions `(N x C x D1 x D2 ... Dn)`, where `N` is the batch size,
  `C` is the number of channels, and `D1`, `D2`, ..., `Dn` are the remaining dimensions.
* `scale` (`@Tensor<T>`) - Scale tensor of shape `(C)`.
* `bias` (`Option<@Tensor<T>>`) - Bias tensor of shape `(C)`. If `None`, no bias is applied.
* `epsilon` (`Option<T>`) (default is zero) - The epsilon value to use to avoid division by zero.

## Panics

* Panics if the scale tensor's shape is not `(C)`.
* Panics if the bias tensor is provided and its shape is not `(C)`.

## Returns

A new tensor `Tensor<T>` with the same shape as the input tensor, after applying instance normalization.

## Example

```rust
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn instance_normalization_example() -> Tensor<FP16x16> {
   
    let epsilon = Option::Some( FixedTrait::new(6554, false));

    let mut shape = ArrayTrait::<usize>::new();
        shape.append(2);
        shape.append(3);
        shape.append(2);
        shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24149, sign: false });
    data.append(FP16x16 { mag: 35894, sign: false });
    data.append(FP16x16 { mag: 38633, sign: true });
    data.append(FP16x16 { mag: 37793, sign: true });
    data.append(FP16x16 { mag: 23838, sign: false });
    data.append(FP16x16 { mag: 5937, sign: false });
    data.append(FP16x16 { mag: 13047, sign: true });
    data.append(FP16x16 { mag: 55527, sign: false });
    data.append(FP16x16 { mag: 97165, sign: true });
    data.append(FP16x16 { mag: 77657, sign: false });
    data.append(FP16x16 { mag: 7142, sign: false });
    data.append(FP16x16 { mag: 96338, sign: false });
    data.append(FP16x16 { mag: 24716, sign: true });
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 41865, sign: true });
    data.append(FP16x16 { mag: 81535, sign: false });
    data.append(FP16x16 { mag: 81322, sign: true });   
    let scale = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 11243, sign: true });
    data.append(FP16x16 { mag: 122125, sign: false });
    data.append(FP16x16 { mag: 97543, sign: false });
    let bias = TensorTrait::new(shape.span(), data.span());

    return X.instance_normalization(@scale,@bias, epsilon);
}
>>>
   [[[[-0.72982788],
    [ 0.38671875]],

   [[ 0.83106995],
     [ 2.89585876]],

   [[ 0.97167969],
    [ 2.00509644]]],

  [[[-0.78804016],
    [ 0.44493103]],

   [[ 2.93608093],
    [ 0.79092407]],

   [[ 0.31443787],
    [ 2.66233826]]]]
``` 
