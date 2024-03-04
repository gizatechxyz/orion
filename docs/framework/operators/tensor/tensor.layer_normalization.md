# tensor.layer_normalization

```rust 
   fn layer_normalization(
    self: @Tensor<T>,
    scale: @Tensor<T>,
    B: Option<@Tensor<T>>,
    axis: Option<i32>,
    epsilon: Option<T>,
    stash_type: Option<usize>,
) -> (Tensor<T>, Tensor<T>, Tensor<T>);
```

Layer normalization of the input, in two stages.
The first stage is standardization, which makes the normalized elements have zero mean and unit variances.
The second stage then scales and shifts the outcome of the first stage 
## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `scale`(`@Tensor<T>,`) - Scale tensor.
* `B`(`Option<@Tensor<T>>`) - Bias tensor. 
* `axis`(`Option<i32>`) (default is -1) - The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back.
* `epsilon`(`Option<T>`) (default is 0) - The epsilon value to use to avoid division by zero.
* `stash_type`(`Option<usize>`) - Precise the computation precision - unused the precision is defined by the type of the tensor.
## Panics

* Panics if condition rank is not equal to 1.

## Returns 

A new normalized tensor`Tensor<T>`.
A tensor containing the mean `Tensor<T>`.
A tensor containing the inverse standard deviation `Tensor<T>`.

## Example

```rust
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn layer_normalization_example() -> (Tensor<FP16x16>, Tensor<FP16x16>, Tensor<FP16x16>) {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 41143, sign: true });
    data.append(FP16x16 { mag: 51803, sign: false });
    data.append(FP16x16 { mag: 113556, sign: false });
    data.append(FP16x16 { mag: 64774, sign: false });
    data.append(FP16x16 { mag: 866, sign: false });
    data.append(FP16x16 { mag: 698, sign: true });
    data.append(FP16x16 { mag: 106500, sign: false });
    data.append(FP16x16 { mag: 98929, sign: false });
    data.append(FP16x16 { mag: 7551, sign: false });
    data.append(FP16x16 { mag: 30689, sign: true });
    data.append(FP16x16 { mag: 38325, sign: false });
    data.append(FP16x16 { mag: 48164, sign: false });
    let X = TensorTrait::new(shape.span(), data.span());

    let shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 49855, sign: false });
    data.append(FP16x16 { mag: 150787, sign: false });
    data.append(FP16x16 { mag: 83498, sign: true });
    data.append(FP16x16 { mag: 30346, sign: false });
    let scale = TensorTrait::new(shape.span(), data.span());

     
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54864, sign: true });
    data.append(FP16x16 { mag: 50952, sign: false });
    data.append(FP16x16 { mag: 8870, sign: true });
    data.append(FP16x16 { mag: 23216, sign: true });
    let bias = TensorTrait::new(shape.span(), data.span());

    return X.layer_normalization(@scale,Option::Some(@bias),Option::None,Option::None,Option::None);
}
>>> [[-0.48926553  1.0185822  -0.02138367 -0.39223218]
     [-0.7945549   0.99696046  0.04332176 -0.412645  ]
     [-0.5664707   0.7491956  -0.7896356  -0.5320859 ]]

``` 
