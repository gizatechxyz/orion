# NNTrait::global_average_pool

```rust
        fn global_average_pool(tensor: @Tensor<T>) -> Tensor<T>;
```

GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same channel. 
This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor.

## Args

* `tensor`(`@Tensor<T>`) - Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.

## Returns

* Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input. The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1.

## Examples

```rust
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::numbers::{FixedTrait, FP8x23};
use orion::operators::nn::NNTrait;
use orion::operators::nn::FP8x23NN;

fn example() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 85392644, sign: false });
    data.append(FP8x23 { mag: 61594092, sign: false });
    data.append(FP8x23 { mag: 163676643, sign: true });
    data.append(FP8x23 { mag: 180530738, sign: false });
    data.append(FP8x23 { mag: 168048412, sign: true });
    data.append(FP8x23 { mag: 5915510, sign: false });
    data.append(FP8x23 { mag: 9047009, sign: true });
    data.append(FP8x23 { mag: 46030420, sign: false });
    data.append(FP8x23 { mag: 184797857, sign: false });
    data.append(FP8x23 { mag: 129370611, sign: false });
    data.append(FP8x23 { mag: 174006060, sign: true });
    data.append(FP8x23 { mag: 162252480, sign: false });
    data.append(FP8x23 { mag: 139240444, sign: true });
    data.append(FP8x23 { mag: 168836878, sign: true });
    data.append(FP8x23 { mag: 246913333, sign: true });
    data.append(FP8x23 { mag: 1047194, sign: true });
    data.append(FP8x23 { mag: 238599466, sign: true });
    data.append(FP8x23 { mag: 216763643, sign: true });
    data.append(FP8x23 { mag: 40581779, sign: true });
    data.append(FP8x23 { mag: 209811161, sign: true });
    data.append(FP8x23 { mag: 250078311, sign: false });
    data.append(FP8x23 { mag: 31811183, sign: true });
    data.append(FP8x23 { mag: 36411415, sign: true });
    data.append(FP8x23 { mag: 107986324, sign: false });
    data.append(FP8x23 { mag: 69727339, sign: false });
    data.append(FP8x23 { mag: 223159880, sign: true });
    data.append(FP8x23 { mag: 184932087, sign: true });
    data.append(FP8x23 { mag: 118617436, sign: false });
    data.append(FP8x23 { mag: 134825391, sign: true });
    data.append(FP8x23 { mag: 217861279, sign: false });
    data.append(FP8x23 { mag: 199069387, sign: false });
    data.append(FP8x23 { mag: 192925915, sign: true });
    let tensor1 = TensorTrait::new(shape.span(), data.span());

    return NNTrait::global_average_pool(@tensor1);
}
>>> [{ mag: 40960207, sign: true } { mag: 31287372, sign: false } { mag: 75603722, sign: true } { mag: 139009462, sign: false } { mag: 176439012, sign: false } { mag: 72460509, sign: true } { mag: 54936798, sign: false } { mag: 22294840, sign: true } ]
```
