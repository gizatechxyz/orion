# TensorTrait::random_uniform_like

```rust
        fn random_uniform_like(tensor: @Tensor<T>, high: Option<T>, low: Option<T>, seed: Option<usize>) -> Tensor<T>;
```

RandomUniformLike generates a tensor with random values using a uniform distribution, matching the shape of the input tensor.

This operation creates a new tensor with the same shape as the input tensor, where each element is initialized with a random value sampled from a uniform distribution.

## Args

* `tensor`(`@Tensor<T>`) - The input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
* `high`(Option<T>) - An optional parameter specifying the upper bound (exclusive) of the uniform distribution. If not provided, defaults to 1.0.
* `low`(Option<T>) - An optional parameter specifying the lower bound (inclusive) of the uniform distribution. If not provided, defaults to 0.0.
* `seed`(Option<usize>) - An optional parameter specifying the seed for the random number generator. If not provided, a random seed will be used.

## Returns

* A `Tensor<T>` with the same shape as the input tensor, filled with random values from a uniform distribution within the specified range.

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
    shape.append(1);
    shape.append(8);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 70016, sign: true });
    data.append(FP8x23 { mag: 57536, sign: false });
    data.append(FP8x23 { mag: 116032, sign: false });
    data.append(FP8x23 { mag: 162944, sign: true });
    data.append(FP8x23 { mag: 43360, sign: false });
    data.append(FP8x23 { mag: 128960, sign: false });
    data.append(FP8x23 { mag: 151808, sign: true });
    data.append(FP8x23 { mag: 28368, sign: false });
    data.append(FP8x23 { mag: 21024, sign: false });
    data.append(FP8x23 { mag: 24992, sign: false });
    data.append(FP8x23 { mag: 125120, sign: true });
    data.append(FP8x23 { mag: 79168, sign: true });
    data.append(FP8x23 { mag: 136960, sign: true });
    data.append(FP8x23 { mag: 10104, sign: true });
    data.append(FP8x23 { mag: 136704, sign: false });
    data.append(FP8x23 { mag: 184960, sign: true });
    let tensor = TensorTrait::new(shape.span(), data.span());
    return TensorTrait::random_uniform_like(@tensor, Option::Some(FP8x23 { mag: 83886080, sign: false }),Option::Some(FP8x23 { mag: 8388608, sign: false }), Option::Some(354145));
}
>>> [[[[7299130, 4884492]], [[2339070, 1559536]], [[3448557, 984617]], [[5745934, 3670947]], [[4665989, 3079292]], [[3375288, 948254]], [[3749966, 4911069]], [[1358829, 4368105]]]]
```
