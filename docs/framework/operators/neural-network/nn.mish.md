# NNTrait::mish

```rust
        fn mish(tensor: @Tensor<T>) -> Tensor<T>;
```

A Self Regularized Non-Monotonic Neural Activation Function.
Perform the linear unit element-wise on the input tensor X using formula:
```rust
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

* A `Tensor<T>` with the same shape as the input tensor.

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
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 29330286, sign: true });
    data.append(FP8x23 { mag: 29576280, sign: false });
    data.append(FP8x23 { mag: 605854, sign: false });
    data.append(FP8x23 { mag: 26167402, sign: false });
    data.append(FP8x23 { mag: 24733382, sign: false });
    data.append(FP8x23 { mag: 5248967, sign: true });
    let tensor1 = TensorTrait::new(shape.span(), data.span());

    return NNTrait::mish(@tensor1);
}
>>> [FP8x23 { mag: 875391, sign: true } , FP8x23 { mag: 29527976, sign: false } , FP8x23 { mag: 377454, sign: false } , FP8x23 { mag: 26073864, sign: false } , FP8x23 { mag: 24610957, sign: false } , FP8x23 { mag: 2120704, sign: true })]
```